import csv
import os
import numpy as np
import h5py

train_data = []
train_label = []
test_data = []
test_label = []

save_path = "./data/data.h5"
if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

with open("emotion.csv", 'r') as input:
    dataset = csv.reader(input)
    for data in dataset:
        if data[-1] == 'Training':
            pixel_list = []
            for pixel in data[1].split(" "):
                pixel_list.append(int(pixel))
            pixel_array = np.asarray(pixel_list)
            train_data.append(pixel_array.tolist())
            train_label.append(int(data[0]))

        if data[-1] == 'Test':
            pixel_list = []
            for pixel in data[1].split(" "):
                pixel_list.append(int(pixel))
            pixel_array = np.asarray(pixel_list)
            test_data.append(pixel_array.tolist())
            test_label.append(int(data[0]))

file = h5py.File(save_path, 'w')
file.create_dataset("train_data", dtype = 'uint8', data = train_data)
file.create_dataset("train_label", dtype = 'int64', data = train_label)
file.create_dataset("test_data", dtype = 'uint8', data = test_data)
file.create_dataset("test_label", dtype = 'int64', data = test_label)
file.close()
print("finish")