## [imports]
import numpy as np
import numpy.random as rnd


## [dataset]
np.random.seed(1337)

# [A, B]
X = [
    [[0], [0]],
    [[1], [0]],
    [[0], [1]],
    [[1], [1]],
]
# Y = X[0] xor X[1]
# [1, 0]:[on, off]
Y = [
    [[0], [1]],
    [[1], [0]],
    [[1], [0]],
    [[0], [1]],
]


## [MPLU]

# Y[w, o, l] = W[w, o, i] * X[i, l] + B[w, o, l]
# 3:
# weights_gradients[w, o, i]
# biases_gradients[w, o, l]
# input_gradients[i, l]


class MPLU:
    def __init__(self, input_size: int, output_size: int, wpi: int) -> None:
        bias_shape = (wpi, output_size, 1)
        weights_shape = (wpi, output_size, input_size)
        self.input = np.zeros((input_size, 1))
        self.weights = rnd.randn(*weights_shape)
        self.biases = rnd.randn(*bias_shape)
        self.wpi = wpi

    def forward(self, input):
        self.input = np.array(input)
        output = np.einsum("il,woi->wol", input, self.weights) + self.biases

        return output

    def backward(self, output_gradients, learning_rate):
        weights_gradients = np.einsum("wol,il->woi", output_gradients, self.input)
        self.weights -= learning_rate * weights_gradients
        self.biases -= learning_rate * output_gradients
        input_gradients = np.einsum("wol,woi->il", output_gradients, self.weights)

        return input_gradients


class Dense:
    def __init__(self, input_size, output_size) -> None:
        self.weights = rnd.randn(output_size, input_size)
        self.biases = rnd.randn(output_size, 1)

    def forward(self, input):
        self.input = np.array(input)
        return np.dot(self.weights, input) + self.biases

    def backward(self, output_gradients, learning_rate):
        weights_gradienst = np.dot(output_gradients, self.input.T)
        biases_gradients = output_gradients
        input_gradients = np.dot(self.weights.T, output_gradients)

        self.weights -= weights_gradienst * learning_rate
        self.biases -= biases_gradients * learning_rate

        return input_gradients


## [activation]
class Activation:
    def __init__(self, act, grad):
        self.act = act
        self.grad = grad

    def forward(self, input):
        self.input = input
        return self.act(input)

    def backward(self, og, _):
        output = np.multiply(og, self.grad(self.input))
        return output


class Tanh(Activation):
    def __init__(self):
        super().__init__(np.tanh, lambda x: 1 - np.tanh(x) ** 2)


class TanhTransformed(Activation):
    def __init__(self):
        super().__init__(lambda x: (1 + np.tanh(x)) / 2, lambda x: (1 - np.tanh(x) ** 2) / 2)


## [summation]
class Sum:
    def __init__(self, wpi) -> None:
        self.wpi = wpi

    def forward(self, input):
        return np.sum(input, axis=0)

    def backward(self, output_gradients, _):
        input_gradients = np.full(shape=(self.wpi, *output_gradients.shape), fill_value=output_gradients)

        return input_gradients


## [training]
def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def MSE_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def BCE(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return -np.mean((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))


def BCE_grad(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return (((1 - y_true) / (1 - y_pred)) - ((y_true) / (y_pred))) / np.size(y_true)


## [utility functions]
def predict(arch, input):
    for l in arch:
        input = l.forward(input)

    return input


def learn(arch, output_grad, learning_rate):
    for l in reversed(arch):
        output_grad = l.backward(output_grad, learning_rate)

    return output_grad


## [training]
epochs = 10_000
learning_rate = 4e-3
wpi1 = 100
model1 = [
    MPLU(2, 2, wpi1),
    Tanh(),
    # TanhTransformed(),
    Sum(wpi1),
]
model2 = [
    Dense(2, 100),
    Tanh(),
    # TanhTransformed(),
    Dense(100, 2),
    Tanh(),
    # TanhTransformed(),
]

for epoch in range(epochs):
    error1 = 0.0
    error2 = 0.0
    for x, y in zip(X, Y):
        prediction1 = predict(model1, x)
        prediction2 = predict(model2, x)
        error1 += MSE(y, prediction1)
        error2 += MSE(y, prediction2)

        grads1 = MSE_grad(y, prediction1)
        grads2 = MSE_grad(y, prediction2)

        learn(model1, grads1, learning_rate)
        learn(model2, grads2, learning_rate)

    error1 /= len(X)
    error2 /= len(X)
    # if (epoch + 1) % 1000 == 0:
    print(
        "=============================================#"
        + f"\nepoch: {epoch+1}/{epochs}"
        + f"\nModel1 Error: {error1}"
        + f"\nModel2 Error: {error2}"
        + "\n=============================================#"
    )


## [testing]
print("Testing Model1")
print(predict(model1, [[0], [0]]))
print(predict(model1, [[0], [1]]))
print(predict(model1, [[1], [0]]))
print(predict(model1, [[1], [1]]))
print("======================")
print("Testing Model2")
print(predict(model2, [[0], [0]]))
print(predict(model2, [[0], [1]]))
print(predict(model2, [[1], [0]]))
print(predict(model2, [[1], [1]]))

## [notes for future improvement]
message = """
while it looks good as it is SAMPU requires more research and advancements and here are some notes for future development
1. adam training.
while the Layer learns fast as it is the Layer sometimes get stuck in some solid state
where it's accuracy is less effective using normal stochastic gradient descent
and I guess that is because the model is falling in local minimum pools

2. better initialization.
I suggest also the use of Xavier Initialization to make it train faster

3. tests on more activations.
testing it on many activations I found that it preforms less better with some activations and here are the results.
- Tanh (BEST)
- Sigmoid (Good)
- Tanh Transformed (Not Good)
- SeLU (Not Bad)
- ReLU (Bad) # doesn't learn any thing at all :\

for activations like Softmax I suggest using an activation like Sigmoid then a Sum process.
"""


#
