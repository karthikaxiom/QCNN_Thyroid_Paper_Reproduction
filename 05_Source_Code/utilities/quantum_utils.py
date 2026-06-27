import pennylane as qml
from pennylane import numpy as np

# QUANTUM DEVICE

N_QUBITS = 4
N_LAYERS = 2

dev = qml.device(
    "default.qubit",
    wires=N_QUBITS
)

# FIXED RANDOM PARAMETERS

np.random.seed(42)

weights = np.random.uniform(
    low=0,
    high=2*np.pi,
    size=(N_LAYERS, N_QUBITS)
)

# QUANTUM FILTER

@qml.qnode(dev)
def quantum_filter(patch):

    # Encoding

    for i in range(N_QUBITS):
        qml.RX(float(patch[i]), wires=i)

    # Random Quantum Circuit

    for layer in range(N_LAYERS):

        for wire in range(N_QUBITS):
            qml.RY(
                weights[layer][wire],
                wires=wire
            )

        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[1,2])
        qml.CNOT(wires=[2,3])
        qml.CNOT(wires=[3,0])

    # Decoding

    return [
        qml.expval(qml.PauliZ(i))
        for i in range(N_QUBITS)
    ]

# TEST

if __name__ == "__main__":

    sample_patch = np.array([
        0.2,
        0.4,
        0.6,
        0.8
    ])

    output = quantum_filter(sample_patch)

    print("="*50)
    print("Quantum Filter Test")
    print("="*50)

    print("Input Patch")
    print(sample_patch)

    print("\nQuantum Output")

    print(output)

    print("\nOutput Length :", len(output))