from qiskit import QuantumCircuit
qasm = """
OPENQASM 2.0;
include "qelib1.inc";
gate rzx(param0) q0,q1 { h q1; cx q0,q1; rz(param0) q1; cx q0,q1; h q1; }
gate ecr q0,q1 { rzx(pi/4) q0,q1; x q0; rzx(-pi/4) q0,q1; }
qreg q[127];
creg c[4];
rz(-pi/2) q[0];
sx q[0];
rz(-0.7312963267948973) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(1.1989) q[1];
sx q[1];
ecr q[1],q[0];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.7941763267948971) q[2];
sx q[2];
rz(pi/2) q[3];
sx q[3];
rz(-1.6740999999999993) q[3];
ecr q[3],q[2];
x q[3];
measure q[1] -> c[1];
measure q[2] -> c[2];

"""
circuit = QuantumCircuit.from_qasm_str(qasm)