![image](https://github.com/robathanjames/mnr-v-quantum_tests/assets/97772344/3c560c2e-f493-470c-940d-a316a5f898b1) ![image](https://github.com/robathanjames/mnr-v-quantum_tests/assets/97772344/d789139c-436b-4894-9f09-30482f711f89)

4-qubit example

The model is trained on Open Images V7 and is being used in other computer vision projects I'm working on currently.

In this case, the input is a 16x16 image and the output is a 256-D feature vector. This vector encapsulates high-level features extracted from the image for the classification task.

A quantum register of 256 qubits is initialized. Each qubit corresponds to a feature in the 256-dimensional vector from the CNN. Every qubit is subjected to a Hadamard gate, placing them in a superposition state. 

The feature vector from the classical model, representing various aspects of the image, are encoded onto the qubits using rotation gates (RY). The angles for the RY gates are directly derived from the CNN's output features.

The algorithm employs controlled-X (CX) gates to introduce entanglement among the qubits. This entanglement allows the system to represent and process correlations between different features of the image.
The final step involves measuring each qubit.

The transpiled circuit is ran and the resulting data is then added to the input of the final layer of the classical model, allowing it to learn the correlation between these features and the labels.

As of right now, I'm defining a loss function that includes a term representing the distance between classical predictions and quantum-enhanced predictions, encouraging the model to learn patterns suggested by quantum results.

I'll see what results come from this. It's been more of a holiday side project than anything else, due to the practicality, if nothing else.
