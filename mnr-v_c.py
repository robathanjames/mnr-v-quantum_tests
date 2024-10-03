import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.visualization import plot_histogram

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Assuming input image size is 28x28

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x

# Quantum Circuit as a Quantum Layer with Parameterized Quantum Circuits
def quantum_layer(feature_vector):
    n_qubits = len(feature_vector)
    params = [Parameter(f'theta_{i}') for i in range(n_qubits)]
    circuit = QuantumCircuit(n_qubits, n_qubits)
    
    # Apply Ry gates with parameterized rotation angles
    for i, param in enumerate(params):
        circuit.ry(param, i)
    
    # Apply entangling gates (CNOT)
    for i in range(n_qubits - 1):
        circuit.cx(i, i + 1)
    
    # Bind parameters to feature vector values
    parameter_binds = {param: feature_vector[i] for i, param in enumerate(params)}
    
    # Measure all qubits
    circuit.measure(range(n_qubits), range(n_qubits))
    
    # Run the quantum circuit on a simulator
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, simulator, shots=1024, parameter_binds=[parameter_binds]).result()
    counts = result.get_counts()
    
    # Convert counts to a feature vector (using expectation values)
    expectation_values = []
    for i in range(n_qubits):
        zeros = sum([counts[state] for state in counts if state[i] == '0'])
        ones = sum([counts[state] for state in counts if state[i] == '1'])
        expectation_values.append((ones - zeros) / (ones + zeros))
    
    return np.array(expectation_values)

# Hybrid Model combining CNN and Quantum Layer
class HybridModel(nn.Module):
    def __init__(self):
        super(HybridModel, self).__init__()
        self.cnn = SimpleCNN()
        self.fc2 = nn.Linear(128, 4)  # Output size must match the number of qubits in quantum layer
        self.fc3 = nn.Linear(4, 10)   # Final classification layer for 10 classes

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc2(x)
        x = x.detach().numpy()  # Convert tensor to numpy array for quantum processing
        quantum_output = quantum_layer(x)  # Pass through quantum layer
        quantum_output = torch.tensor(quantum_output, dtype=torch.float32)  # Convert back to tensor
        x = F.relu(quantum_output)
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
