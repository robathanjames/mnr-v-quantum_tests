from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Gate
from qiskit.circuit.library.standard_gates import RZGate, SXGate, CXGate, HGate, XGate
import numpy as np

# Define the rzx custom gate
def rzx(theta):
    qc = QuantumCircuit(2, name='rzx')
    qc.h(1)
    qc.cx(0, 1)
    qc.rz(theta, 1)
    qc.cx(0, 1)
    qc.h(1)
    return qc.to_gate()

# Define the ecr custom gate using rzx
def ecr():
    pi = np.pi
    rzx_pi_over_4 = rzx(pi / 4)
    rzx_neg_pi_over_4 = rzx(-pi / 4)
    qc = QuantumCircuit(2, name='ecr')
    qc.append(rzx_pi_over_4, [0, 1])
    qc.x(0)
    qc.append(rzx_neg_pi_over_4, [0, 1])
    return qc.to_gate()

# Initialize quantum and classical registers
qr = QuantumRegister(4, 'q')
cr = ClassicalRegister(4, 'c')
circuit = QuantumCircuit(qr, cr)

# Define constants
pi = np.pi

# Apply gates to q[0]
circuit.rz(-pi/2, qr[0])
circuit.sx(qr[0])
circuit.rz(-0.7312963267948973, qr[0])
circuit.sx(qr[0])
circuit.rz(pi/2, qr[0])

# Apply gates to q[1]
circuit.rz(-pi/2, qr[1])
circuit.sx(qr[1])
circuit.rz(1.1989, qr[1])
circuit.sx(qr[1])

# Apply ECR gate between q[1] and q[0]
circuit.append(ecr(), [qr[1], qr[0]])

# Continue with q[0] and q[1]
circuit.rz(-pi/2, qr[0])
circuit.sx(qr[0])
circuit.rz(pi/2, qr[0])

circuit.rz(pi/2, qr[1])
circuit.sx(qr[1])
circuit.rz(pi/2, qr[1])

# Apply gates to q[2]
circuit.rz(-pi/2, qr[2])
circuit.sx(qr[2])
circuit.rz(1.7941763267948971, qr[2])
circuit.sx(qr[2])

# Apply gates to q[3]
circuit.rz(pi/2, qr[3])
circuit.sx(qr[3])
circuit.rz(-1.6740999999999993, qr[3])

# Apply ECR gate between q[3] and q[2]
circuit.append(ecr(), [qr[3], qr[2]])

# Apply X gate to q[3]
circuit.x(qr[3])

# Measure q[1] and q[2]
circuit.measure(qr[1], cr[1])
circuit.measure(qr[2], cr[2])

# Optional: Draw the circuit
print(circuit.draw(output='text'))

# If you need to convert to QASM string later
# qasm_str = circuit.qasm()
