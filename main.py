import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import neuralnetwork as nn
import math

brain = nn.NeuralNetwork(2,2,1)
input = [1,0]
output = brain.feedforward(input,lambda x: 1 / (1 + math.exp(-x)))
print(output)