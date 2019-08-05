import neuralnetwork as nn
import math
import json
import random
with open('training_data.json') as json_file:
    training_data = json.load(json_file)
func = {'activation':lambda x : 1 / (1 + math.exp(-x)),'deactivation':lambda y : y * (1 - y)}
brain = nn.NeuralNetwork(2,10,1,func)

for x in range(50000):
    test_case = random.choice(list(training_data.items()))[1]
    brain.train(test_case['inputs'],test_case['targets'])


print(brain.feedforward([0,0]))
print(brain.feedforward([1,0]))
print(brain.feedforward([0,1]))
print(brain.feedforward([1,1]))


