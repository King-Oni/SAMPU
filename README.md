# Sum Activations Multi Perceptron Unit

SAMPU is an advancement of the Multi Perceptron Layer Unit (MPLU) proposed previously. Which was designed to replace the traditional Multilayer Perceptron (MLP) by offering faster more parallelizable learning as the model scales.

## Overview

SAMPU advances over MPLU by Activating MPLU output before summing them thus turning it from the activation of sums to a sum of activations `σ(Σ(W•X + B))` -> `Σ(σ(W•X+B))`.
By using the collective effort of multiple activated perceptrons the model can scale to large sizes without being that much computationally intensive.

## Key Features

- **Efficient Learning**: SAMPU demonstrates faster learning compared to traditional MLPs, especially as the network scales. as it can be all parallelized in calculation then activated and summed
- **Scalability**: With its novel summation-based architecture, SAMPU is highly scalable, making it suitable for various applications and network sizes.
- **Modularity**: SAMPU's modular design allows for easy integration into existing neural network frameworks and architectures.
- **Flexibility**: SAMPU supports various activation functions, enabling flexibility in network design and experimentation.

## Usage

```python
print(predict(model1, [[0], [0]]))
print(predict(model1, [[0], [1]]))
print(predict(model1, [[1], [0]]))
print(predict(model1, [[1], [1]]))
```

## Contributing

Contributions to SAMPU are more than welcomed! Whether you'd like to report a bug, propose a feature, or submit a pull request, please feel free to contribute to the development of SAMPU.

## License

SAMPU is released under the MIT License. See the [LICENSE](LICENSE) file for more details.
