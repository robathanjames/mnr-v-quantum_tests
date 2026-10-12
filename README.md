# MNR-V: hybrid CNN / quantum experiment

[![CI](https://github.com/robathanjames/mnr-v-quantum_tests/actions/workflows/ci.yml/badge.svg)](https://github.com/robathanjames/mnr-v-quantum_tests/actions/workflows/ci.yml)

A small PyTorch CNN feeds image features into a Qiskit circuit, then classifies
its Pauli-Z expectation values. This runnable research prototype explores hybrid
feature transformations; it does not establish quantum advantage or report
accuracy on a real dataset.

The original four-qubit experiment is preserved, with working batches,
first-order gradients through the circuit, and Qiskit 2.x APIs.

## Quick start

Requires Python 3.10 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python mnr-v_c.py
```

On Windows, activate with `.venv\Scripts\activate` instead. To select a CPU-only
PyTorch wheel, install it before the project:

```bash
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

The default demo performs inference on four synthetic grayscale images containing
vertical or horizontal bars. The demo's two-class classifier is randomly
initialized; predictions before training are not meaningful. No dataset is
downloaded, and no weights or results are written to disk.

```bash
# Train on the same small synthetic batch for five optimizer steps.
python mnr-v_c.py --steps 5 --batch-size 4 --image-size 16 --qubits 2

# Equivalent installed/module entry points; JSON is available for scripting.
mnr-quantum --steps 5 --json
python -m mnr_quantum --batch-size 2 --json
```

The bars demo checks execution and gradients. It is not a generalization
benchmark; that requires a real dataset with independent training/validation/test
splits.

## Architecture and shapes

| Stage | Default output | What happens |
| --- | --- | --- |
| Images | `(B, 1, H, W)` | Floating-point grayscale inputs; `H, W >= 4` |
| CNN | `(B, 128)` | Two convolution/pooling blocks, adaptive pooling, dense features |
| Angle projection | `(B, 4)` | Linear projection and `pi * tanh` bound the RY angles |
| Circuit | `(B, 4)` | One RY per qubit, followed by a nearest-neighbor CNOT chain |
| Classifier | `(B, 10)` | Linear readout returns logits for cross-entropy |

The model defaults to 10 classes, matching the original script. The CLI uses 2
classes for its bars example. Qubit count is configurable from 1 to 8; channels
and class count are configurable through the API. Both 16x16 and 28x28 inputs
work, including rectangular images.

Each image gets its own circuit. Qubits start in `|0>` and receive RY rotations;
there is no initial Hadamard layer. Readout returns
`Z_i = P(bit_i=0) - P(bit_i=1)` in qubit order `0..N-1`. Qiskit places qubit 0 at
the **right** of a counts bitstring. Negative expectation values are retained.

## Train with PyTorch

```python
import torch
from mnr_quantum import HybridModel

model = HybridModel(n_qubits=4, num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
images = torch.rand(3, 1, 28, 28)  # Replace with preprocessed real images.
labels = torch.tensor([2, 7, 1])  # Replace with real class labels.

optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = torch.nn.functional.cross_entropy(logits, labels)
loss.backward()  # Gradients reach the CNN, projection, and classifier.
optimizer.step()
```

Use `logits.softmax(dim=1)` to display probabilities. Cross-entropy expects raw
logits. The circuit has a fixed topology and no separate trainable weights; the
CNN learns which angles to encode.

The custom PyTorch bridge uses the parameter-shift rule for each RY gate:

```text
d f / d theta_j = [f(theta + (pi/2) e_j) - f(theta - (pi/2) e_j)] / 2
```

Simulation runs on the CPU; outputs and gradients retain the input device and
floating dtype. Moving the model to CUDA/MPS does not move the simulator there.
Only first-order differentiation is supported.

Use `QuantumLayer(4)` directly with a floating tensor of shape `(B, 4)` for an
autograd-compatible readout. The original `quantum_layer(array)` NumPy helper
accepts `(4,)` or `(B, 4)` inputs for evaluation without autograd. `SimpleCNN`,
`HybridModel`, and `quantum_layer` remain importable from the original script as
well as the package. Existing prototype checkpoints need retraining: adaptive
pooling changes the first dense layer's input size.

## Exact and finite-shot simulation

**Exact statevector** execution is the default: deterministic, without
measurement noise, and requiring only base Qiskit.

**Finite-shot Aer** execution is optional:

```bash
python -m pip install -e '.[aer]'
python mnr-v_c.py --shots 1024 --seed 7 --batch-size 2
```

In Python, use `HybridModel(shots=1024, seed=7)` or
`QuantumLayer(4, shots=1024, seed=7)`. Forward values and parameter-shift gradients
both use the sampled backend and have sampling error. Increase shots to reduce
that error. A fixed seed makes repeated evaluations reproducible with the same
software/platform; pass `seed=None` through the API to leave it unset. Exact mode
is recommended when debugging training.

The parameterized circuit is built once per layer; Aer transpilation also happens
once. Each batch, including shifted circuits during backward, is submitted in one
backend job. No IBM account, hardware connection, or credentials are needed.
Statevector memory scales as `2**N`; backward adds `2*N` circuit evaluations per
image. The eight-qubit limit keeps this prototype's scope small.

For this RY/CNOT chain, exact outputs have a simple classical formula:
`Z_i = product(cos(theta_j), j=0..i)`. Tests use this independent expression to
verify readout and gradients. Entanglement alone does not imply a computational
or classification advantage.

## Development

```bash
python -m pip install -e '.[aer,dev]'
pytest
ruff check .
ruff format --check .
python -m build
```

Tests cover analytic expectations, bit order and sign, independent batches,
finite-difference gradient checking, updates to every classical block,
dtype/device handling, checkpoint round trips, sampled execution, reproducible
training, and invalid configuration. Aer tests skip without the optional
dependency; CUDA coverage skips when unavailable. GitHub Actions runs CPU checks
on Python 3.10, 3.12, and 3.13.

## Research status

Earlier notes described Open Images V7 training and a 256-qubit design. This
repository does not include that dataset pipeline, trained weights, or a measured
real-data result. The implemented encoder has 128 features, compressed to four
rotation angles by default. The synthetic example makes this smaller experiment
reproducible before extending it to real data. A classical/quantum agreement loss
is not implemented.

Useful next experiments are a parameter-matched classical baseline, multiple
random seeds, held-out accuracy, runtime measurements, and a sweep over qubit
count and shot budget.

References: [Statevector API](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.quantum_info.Statevector),
[Qiskit bit ordering](https://quantum.cloud.ibm.com/docs/en/guides/bit-ordering),
[AerSimulator API](https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.AerSimulator.html),
and [parameter-shift gradients](https://arxiv.org/abs/1905.13311).
