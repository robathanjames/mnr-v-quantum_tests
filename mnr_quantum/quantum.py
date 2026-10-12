"""RY encoding and a CNOT chain, with exact or sampled Pauli-Z readout."""

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import torch
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from torch import nn
from torch.autograd.function import once_differentiable

MAX_QUBITS = 8


def _integer(name: str, value: int, minimum: int = 1, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return int(value)


def counts_to_expectations(counts: Mapping[str, int], n_qubits: int) -> np.ndarray:
    """Return P(0)-P(1) for qubits 0..N-1, respecting Qiskit's little-endian keys.

    Spaces separating classical registers are accepted. Counts must use binary
    bitstrings and nonnegative integer frequencies with a positive total.
    """
    n_qubits = _integer("n_qubits", n_qubits, maximum=MAX_QUBITS)
    totals = np.zeros(n_qubits, dtype=np.float64)
    shots = 0
    for key, count in counts.items():
        if not isinstance(key, str):
            raise ValueError("count keys must be binary bitstrings")
        bits = key.replace(" ", "")
        if len(bits) != n_qubits or any(bit not in "01" for bit in bits):
            raise ValueError(f"count keys must contain exactly {n_qubits} binary digits")
        count = _integer("count", count, minimum=0)
        for qubit, bit in enumerate(reversed(bits)):
            totals[qubit] += count if bit == "0" else -count
        shots += count
    if shots == 0:
        raise ValueError("counts must contain at least one shot")
    return totals / shots


class _CircuitRunner:
    """Build a template once and evaluate independent circuits for each sample."""

    def __init__(self, n_qubits: int, shots: int | None, seed: int | None):
        self.n_qubits = _integer("n_qubits", n_qubits, maximum=MAX_QUBITS)
        self.shots = None if shots is None else _integer("shots", shots)
        self.seed = None if seed is None else _integer("seed", seed, minimum=0, maximum=2**32 - 1)
        self.parameters = ParameterVector("theta", self.n_qubits)
        self.circuit = QuantumCircuit(self.n_qubits)
        for qubit, parameter in enumerate(self.parameters):
            self.circuit.ry(parameter, qubit)
        for qubit in range(self.n_qubits - 1):
            self.circuit.cx(qubit, qubit + 1)

        # A basis index's least significant bit is qubit 0 in Qiskit.
        basis = np.arange(2**self.n_qubits)[:, None]
        self._z_signs = 1 - 2 * ((basis >> np.arange(self.n_qubits)) & 1)
        self._backend = None
        self._measured_circuit = None
        if self.shots is not None:
            try:
                from qiskit_aer import AerSimulator
            except ImportError as error:
                raise ImportError(
                    "Sampled execution requires Qiskit Aer. "
                    "Install with: pip install 'mnr-v-quantum-tests[aer]' "
                    "(or pip install -e '.[aer]' from the repository)."
                ) from error
            self._backend = AerSimulator(method="statevector", max_parallel_threads=1)
            measured = self.circuit.copy()
            measured.measure_all()
            self._measured_circuit = transpile(
                measured, self._backend, seed_transpiler=self.seed, optimization_level=0
            )

    def evaluate(self, angles: np.ndarray) -> np.ndarray:
        if angles.ndim != 2 or angles.shape[1] != self.n_qubits or angles.shape[0] == 0:
            raise ValueError(f"angles must have shape (nonempty batch, {self.n_qubits})")
        if not np.isfinite(angles).all():
            raise ValueError("angles must be finite")
        if self.shots is None:
            outputs = []
            for sample in angles:
                bound = self.circuit.assign_parameters(
                    dict(zip(self.parameters, sample, strict=True))
                )
                probabilities = Statevector.from_instruction(bound).probabilities()
                outputs.append(probabilities @ self._z_signs)
            return np.clip(np.stack(outputs), -1.0, 1.0)

        circuits = [
            self._measured_circuit.assign_parameters(
                dict(zip(self.parameters, sample, strict=True))
            )
            for sample in angles
        ]
        result = self._backend.run(circuits, shots=self.shots, seed_simulator=self.seed).result()
        return np.stack(
            [
                counts_to_expectations(result.get_counts(i), self.n_qubits)
                for i in range(len(circuits))
            ]
        )

    def jacobian(self, angles: np.ndarray) -> np.ndarray:
        """Return (batch, input angle, output qubit) derivatives by parameter shift."""
        batch = angles.shape[0]
        shifted = np.repeat(angles[:, None, :], 2 * self.n_qubits, axis=1)
        for qubit in range(self.n_qubits):
            shifted[:, 2 * qubit, qubit] += np.pi / 2
            shifted[:, 2 * qubit + 1, qubit] -= np.pi / 2
        values = self.evaluate(shifted.reshape(-1, self.n_qubits))
        values = values.reshape(batch, self.n_qubits, 2, self.n_qubits)
        return (values[:, :, 0, :] - values[:, :, 1, :]) / 2


class _QuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles: torch.Tensor, runner: _CircuitRunner) -> torch.Tensor:
        ctx.runner = runner
        ctx.save_for_backward(angles)
        values = angles.detach().to(device="cpu", dtype=torch.float64).numpy()
        return angles.new_tensor(runner.evaluate(values))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        (angles,) = ctx.saved_tensors
        values = angles.detach().to(device="cpu", dtype=torch.float64).numpy()
        upstream = grad_output.detach().to(device="cpu", dtype=torch.float64).numpy()
        gradients = np.einsum("bpo,bo->bp", ctx.runner.jacobian(values), upstream)
        return angles.new_tensor(gradients), None


class QuantumLayer(nn.Module):
    """Map (batch, qubits) rotation angles to signed Z expectation values.

    Exact statevector execution is the default. Set ``shots`` to use Aer and
    sampled parameter-shift gradients. Simulation runs on the CPU; outputs and
    first-order gradients retain the input tensor's device and floating dtype.
    The fixed circuit has no separate trainable weights.
    """

    def __init__(self, n_qubits: int = 4, *, shots: int | None = None, seed: int | None = 7):
        super().__init__()
        self._runner = _CircuitRunner(n_qubits, shots, seed)
        self.n_qubits = self._runner.n_qubits

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        if not isinstance(angles, torch.Tensor) or not torch.is_floating_point(angles):
            raise TypeError("angles must be a floating-point torch.Tensor")
        if angles.ndim != 2 or angles.shape[0] == 0 or angles.shape[1] != self.n_qubits:
            raise ValueError(f"angles must have shape (nonempty batch, {self.n_qubits})")
        return _QuantumFunction.apply(angles, self._runner)

    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}, shots={self._runner.shots}, seed={self._runner.seed}"


def quantum_layer(feature_vector, *, shots: int | None = None, seed: int | None = 7) -> np.ndarray:
    """Evaluate a NumPy vector or batch without autograd; use QuantumLayer to train."""
    angles = np.asarray(feature_vector, dtype=np.float64)
    if angles.ndim not in (1, 2):
        raise ValueError("feature_vector must be a vector or a batch of vectors")
    single = angles.ndim == 1
    if single:
        angles = angles[None, :]
    runner = _CircuitRunner(angles.shape[1], shots, seed)
    values = runner.evaluate(angles)
    return values[0] if single else values
