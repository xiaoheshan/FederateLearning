import numpy as np

from layer import Layer


class LinearLayer(Layer):
    def __init__(self, w, b):
        self.w = np.array(w)
        self.b = np.array(b)

    def simple_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        return x.dot(self.w) + self.b

    def backward(self, learning_rate, y, x, *arg):
        output_grad = arg[0]
        input_grad = output_grad.dot(self.w.T)
        dw = x.T.dot(output_grad)
        db = np.sum(output_grad, axis=0)
        self.w = self.w - learning_rate * dw
        self.b = self.b - learning_rate * db
        return input_grad

    def explore(self, sigma):
        self.w += np.random.normal(0, np.max(self.w) * sigma, self.w.shape)
        self.b += np.random.normal(0, np.max(self.b) * sigma, self.b.shape)

    def mutate(self, mutation_probability):
        self.w = self.mutate_array(self.w, mutation_probability)
        self.b = self.mutate_array(self.b, mutation_probability)


class ReluLayer(Layer):
    def simple_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        return np.maximum(x, 0.)

    def backward(self, learning_rate, y, x, *arg):
        return np.multiply(np.greater(y, 0.), arg[0])


class SoftmaxOutputLayer(Layer):
    def simple_forward(self, x):
        return self.forward(x)

    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self, learning_rate, y, x, *arg):
        t = arg[0]
        return y - t
