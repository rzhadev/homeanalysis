import neuralnetwork as nn
import matplotlib.pyplot as plt
import numpy as np
import math

training_size = 60000
training_images = np.load(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\train\train_images.npy')
training_labels = np.load(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\train\train_labels.npy')
testing_images = np.load(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\test\test_images.npy')
testing_labels = np.load(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\test\test_labels.npy')
brain = nn.NeuralNetwork(784,20,10,{'activation':lambda x: 1 / (1 + math.exp(-x)),'deactivation':lambda y : y * (1 - y)})
for x in range(5):
    for i in range(60000):
        targets = [0,0,0,0,0,0,0,0,0,0]
        targets[training_labels[i]] = 1
        print(training_images[i])
        print(training_labels[i])
        brain.train(training_images[i],targets)
    #np.random.shuffle(training_images)
    print(f"{x} epoch completed")
correct = 0 
for x in range(0,10000):
    label = testing_labels[x]
    print(label)
    print(brain.feedforward(testing_images[x]))
    if brain.feedforward(testing_images[x]) == label:
        correct += 1
print(correct/10000)





