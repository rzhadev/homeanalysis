import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import neuralnetwork as nn
import math

brain = nn.NeuralNetwork(2,2,2)
input = [1,0]
targets = [1,1]
output = brain.train(input,targets,lambda x: 1 / (1 + math.exp(-x)))