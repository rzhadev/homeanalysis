import numpy as np
import struct as st 
import json 
with open(r'filelocations.json') as f:
    locations = json.load(f)
'''
    -load dataset from file
    -unpack data into numpy array
    -convert to scale of 0-1 instead of 0-255
'''
print("formatting training images...")
train_data = open(locations["raw-train-images"],'rb')
train_data.seek(0)
magic,size = st.unpack('>II',train_data.read(8))
rows,cols = st.unpack('>II',train_data.read(8))
data = np.fromfile(train_data, dtype=np.dtype(np.uint8).newbyteorder('>'))
data = np.divide((255 - data), 255)
data = data.reshape((size,rows,cols))
np.save(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\train\train_images.npy',data)
#save train data to file
print("formatting training labels...")
train_label = open(locations["raw-train-labels"],'rb')
train_label.seek(0)
magic,size = st.unpack('>II',train_label.read(8))
data_label = np.fromfile(train_label,dtype=np.dtype(np.uint8).newbyteorder('>'))
np.save(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\train\train_labels.npy',data_label)
#save train labels to file
print("formatting test images...")
test_data = open(locations["raw-test-images"],'rb')
test_data.seek(0)
magic,size = st.unpack('>II',test_data.read(8))
rows,cols = st.unpack('>II',test_data.read(8))
data = np.fromfile(test_data, dtype=np.dtype(np.uint8).newbyteorder('>'))
data = np.divide((255 - data), 255)
data = data.reshape((size,rows,cols))
np.save(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\test\test_images.npy',data)
#save test data to file 
print("formatting test labels...")
test_label = open(locations["raw-test-labels"],'rb')
test_label.seek(0)
magic,size = st.unpack('>II',test_label.read(8))
data_label = np.fromfile(test_label,dtype=np.dtype(np.uint8).newbyteorder('>'))
np.save(r'C:\Users\Richard Zha\Documents\toyneuralnet.py\data\mnist\test\test_labels.npy',data_label)
#save test lavels to file
print("finished.")
