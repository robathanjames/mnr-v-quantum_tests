import builtins

import numpy as np
import pytest
import torch

from mnr_quantum import QuantumLayer, counts_to_expectations, quantum_layer


def test_cnot_chain_matches_analytic_expectations():
    angles = np.array([[0.2, -0.7, 1.1, 2.3], [2.1, 0.6, -0.4, 0.9]])
    # Independent formula: <Z_i> = product_{j=0..i} cos(theta_j).
    np.testing.assert_allclose(
        quantum_layer(angles), np.cumprod(np.cos(angles), axis=1), atol=1e-12
    )


def test_single_vector_and_batch_agree():
    angles = np.array([[0.2, 1.0], [0.5, -0.8], [2.0, 0.0]])
    batch = quantum_layer(angles)
    assert batch.shape == (3, 2)
    np.testing.assert_allclose(batch, np.stack([quantum_layer(row) for row in angles]))


def test_z_sign_and_little_endian_counts():
    np.testing.assert_array_equal(counts_to_expectations({"01": 4}, 2), [-1, 1])
    np.testing.assert_allclose(counts_to_expectations({"0 1": 3, "10": 1}, 2), [-0.5, 0.5])
    np.testing.assert_allclose(quantum_layer([0.0, np.pi]), [1.0, -1.0], atol=1e-12)


def test_parameter_shift_passes_finite_difference_gradcheck():
    angles = torch.tensor([[0.3, -0.6, 1.2], [0.8, 0.2, -0.4]], dtype=torch.float64)
    angles.requires_grad_()
    assert torch.autograd.gradcheck(QuantumLayer(3), (angles,), atol=1e-6, rtol=1e-5)


def test_weighted_gradients_match_independent_analytic_formula():
    angles = torch.tensor([[0.3, -0.6, 1.2], [0.8, 0.2, -0.4]], dtype=torch.float64)
    angles.requires_grad_()
    weights = torch.tensor([[0.7, -0.2, 1.1], [-0.5, 0.9, 0.3]], dtype=torch.float64)
    (QuantumLayer(3)(angles) * weights).sum().backward()
    reference = angles.detach().clone().requires_grad_()
    (torch.cos(reference).cumprod(dim=1) * weights).sum().backward()
    torch.testing.assert_close(angles.grad, reference.grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_output_and_gradient_preserve_dtype(dtype):
    angles = torch.tensor([[0.4, 0.6]], dtype=dtype, requires_grad=True)
    output = QuantumLayer(2)(angles)
    output.sum().backward()
    assert output.dtype == angles.grad.dtype == dtype
    assert output.device == angles.grad.device == angles.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_input_retains_device_and_backpropagates():
    angles = torch.tensor([[0.4, 0.6]], device="cuda", requires_grad=True)
    output = QuantumLayer(2)(angles)
    output.sum().backward()
    assert output.device == angles.grad.device == angles.device
    assert torch.isfinite(angles.grad).all()


@pytest.mark.parametrize("n_qubits", [0, 9, 4.5, True])
def test_invalid_qubit_counts(n_qubits):
    with pytest.raises(ValueError, match="n_qubits"):
        QuantumLayer(n_qubits)


@pytest.mark.parametrize("shots", [0, -1, 1.5, True])
def test_invalid_shots(shots):
    with pytest.raises(ValueError, match="shots"):
        QuantumLayer(shots=shots)


@pytest.mark.parametrize("seed", [-1, 2**32, 0.5, True])
def test_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        QuantumLayer(seed=seed)


@pytest.mark.parametrize("values", [torch.zeros(4), torch.zeros(2, 3), torch.zeros(0, 4)])
def test_rejects_invalid_tensor_shape(values):
    with pytest.raises(ValueError, match="shape"):
        QuantumLayer()(values)


@pytest.mark.parametrize("values", [torch.ones(1, 4, dtype=torch.int64), [[0.0] * 4]])
def test_requires_floating_tensor(values):
    with pytest.raises(TypeError, match="floating-point"):
        QuantumLayer()(values)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_nonfinite_angles(value):
    with pytest.raises(ValueError, match="finite"):
        QuantumLayer(1)(torch.tensor([[value]]))


@pytest.mark.parametrize("values", [[], np.zeros((0, 2)), np.zeros((1, 2, 3))])
def test_numpy_helper_rejects_invalid_inputs(values):
    with pytest.raises(ValueError):
        quantum_layer(values)


@pytest.mark.parametrize("counts", [{}, {"01": 0}, {"01": -1}, {"02": 1}, {"1": 1}, {"01": 1.5}])
def test_invalid_measurement_counts(counts):
    with pytest.raises(ValueError):
        counts_to_expectations(counts, 2)


def test_shot_mode_respects_batch_and_bit_order():
    pytest.importorskip("qiskit_aer")
    angles = np.array([[0.0, np.pi], [np.pi, 0.0], [0.0, 0.0]])
    np.testing.assert_array_equal(quantum_layer(angles, shots=32), [[1, -1], [-1, -1], [1, 1]])


def test_seeded_shot_mode_approximates_exact_values():
    pytest.importorskip("qiskit_aer")
    angles = np.array([[0.4, 1.2], [1.0, -0.2]])
    first = quantum_layer(angles, shots=8192, seed=19)
    np.testing.assert_array_equal(first, quantum_layer(angles, shots=8192, seed=19))
    np.testing.assert_allclose(first, quantum_layer(angles), atol=0.04)


def test_shot_parameter_shift_approximates_analytic_gradient():
    pytest.importorskip("qiskit_aer")
    angles = torch.tensor([[0.4, 1.2]], dtype=torch.float64, requires_grad=True)
    QuantumLayer(2, shots=8192, seed=19)(angles).sum().backward()
    reference = angles.detach().clone().requires_grad_()
    torch.cos(reference).cumprod(dim=1).sum().backward()
    torch.testing.assert_close(angles.grad, reference.grad, atol=0.04, rtol=0)


def test_exact_mode_works_without_aer_and_sampled_mode_explains_install(monkeypatch):
    original_import = builtins.__import__

    def without_aer(name, *args, **kwargs):
        if name == "qiskit_aer":
            raise ImportError("Aer is not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_aer)
    np.testing.assert_array_equal(quantum_layer([0.0]), [1.0])
    with pytest.raises(ImportError, match="pip install"):
        QuantumLayer(1, shots=32)
