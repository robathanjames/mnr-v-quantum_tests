import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit.circuit import Parameter
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.providers.aer import AerSimulator

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Assuming input image size is 28x28
        self.fc2 = nn.Linear(128, 4)  # Output size must match the number of qubits in quantum layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

# Quantum Circuit as a Quantum Layer with Parameterized Quantum Circuits
def create_quantum_circuit(n_qubits):
    params = [Parameter(f'theta_{i}') for i in range(n_qubits)]
    circuit = QuantumCircuit(n_qubits)
    
    # Apply Ry gates with parameterized rotation angles
    for i, param in enumerate(params):
        circuit.ry(param, i)
    
    # Apply entangling gates (CNOT)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    
    return circuit, params

# Create a Quantum Neural Network using Qiskit Machine Learning
def create_quantum_neural_network(n_qubits):
    quantum_circuit, params = create_quantum_circuit(n_qubits)
    backend = AerSimulator()
    quantum_instance = QuantumInstance(backend, shots=1024)
    qnn = EstimatorQNN(quantum_circuit, params, input_params=params, quantum_instance=quantum_instance)
    return qnn

# Hybrid Model combining CNN and Quantum Layer
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.cnn = SimpleCNN()
        self.qnn = create_quantum_neural_network(4)
        self.quantum_layer = TorchConnector(self.qnn)
        self.fc3 = nn.Linear(4, 10)   # Final classification layer for 10 classes

    def forward(self, x):
        x = self.cnn(x)
        x = self.quantum_layer(x)  # Pass through quantum layer
        x = F.relu(x)
        x = self.fc3(x)
        return x

# Example usage
if __name__ == "__main__":
    # Create an instance of the hybrid model
    model = HybridModel()
    
    # Define a dummy input (batch size of 1, 1 channel, 28x28 image)
    input_data = torch.randn(1, 1, 28, 28)
    
    # Perform a forward pass
    output = model(input_data)
    print(output)
