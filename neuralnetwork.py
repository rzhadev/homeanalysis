import numpy as np
import math
class NeuralNetwork:
    def __init__(self, input, hidden, output):
        self.inputNodes = input
        self.hiddenNodes = hidden
        self.outputNodes = output
        self.ih_weights = np.random.random(size=(self.hiddenNodes, self.inputNodes))*2 - 1
        self.oh_weights = np.random.random(size=(self.outputNodes, self.hiddenNodes))*2 - 1
        self.h_bias = np.random.random()*2-1
        self.o_bias = np.random.random()*2-1

    def feedforward(self, input, activation):
        inputs = np.array(input)
        hidden = self.ih_weights @ inputs
        hidden += self.h_bias
        # activation function
        hidden = np.array(list(map(activation, hidden)))
        output = self.oh_weights @ hidden
        output += self.o_bias
        output = np.array(list(map(activation, output)))
        return output

    def train(self,inputs,targets,activation):
        targets = np.array(targets)
        outputs = self.feedforward(inputs,activation)
        error = targets - outputs
        oh_weight_transpose = np.transpose(self.oh_weights)
        hidden_error = oh_weight_transpose @ error