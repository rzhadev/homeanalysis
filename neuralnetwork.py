import numpy as np
import math
class NeuralNetwork:
    def __init__(self, input, hidden, output, func):
        self.inputNodes = input
        self.hiddenNodes = hidden
        self.outputNodes = output
        self.ih_weights = np.random.random(size=(self.hiddenNodes, self.inputNodes))*2 - 1
        self.oh_weights = np.random.random(size=(self.outputNodes, self.hiddenNodes))*2 - 1
        self.h_bias = np.random.random()*2-1
        self.o_bias = np.random.random()*2-1
        self.alpha = 0.1
        self.activation = func['activation']
        self.deactivation = func['deactivation']

    def feedforward(self, inputs):
        inputs = np.reshape(np.asmatrix(inputs),(-1,1))
        hidden = np.matmul(self.ih_weights,inputs)
        hidden = np.add(hidden,self.h_bias)
        #apply activation function
        hidden = np.reshape(np.array(list(map(self.activation, hidden))),(hidden.shape))
        output = np.matmul(self.oh_weights,hidden)
        output = np.add(output,self.o_bias)
        output = np.reshape(np.array(list(map(self.activation, output))),(output.shape))
        return output.tolist()
    def train(self,inputs,targets):
        #feedforward
        inputs = np.reshape(np.asmatrix(inputs),(-1,1))
        hidden = np.matmul(self.ih_weights,inputs)
        hidden = np.add(hidden,self.h_bias)
        hidden = np.reshape(np.array(list(map(self.activation, hidden))),(hidden.shape))
        output = np.matmul(self.oh_weights,hidden)
        output = np.add(output,self.o_bias)
        output = np.reshape(np.array(list(map(self.activation, output))),(output.shape))
        
        #back propagation and gradient descent
        targets = np.asmatrix(targets)
        error = targets - output
        gradient = np.reshape(np.array(list(map(self.deactivation, output))),(output.shape))
        gradient *= error
        gradient *= self.alpha
        hidden_t = np.transpose(hidden)
        #calculate output weight matrix delta
        oh_weight_delta = np.matmul(gradient,hidden_t)
        self.oh_weights = np.add(self.oh_weights,oh_weight_delta)
        self.o_bias = np.add(self.o_bias,gradient)
        #calculate hidden weight matrix delta
        oh_weight_t = np.transpose(self.oh_weights)
        hidden_error = np.matmul(oh_weight_t,error)

        hidden_gradient = np.reshape(np.array(list(map(self.deactivation, hidden))),(hidden.shape))
        hidden_gradient *= hidden_error
        hidden_gradient *= self.alpha

        input_t = np.transpose(inputs)
        ih_weight_delta = np.matmul(hidden_gradient,input_t)
        self.ih_weights = np.add(self.ih_weights,ih_weight_delta)
        self.h_bias = np.add(self.h_bias,hidden_gradient)




       
        

       