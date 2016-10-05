from __future__ import print_function
from imagenet_utils import preprocess_input
from resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
import os.path
import csv

X_Data = np.empty([0,224,224,3])
Y_Data = np.empty([0])

file_idx = 0
img_path = 'data/%d.tif' % file_idx
csv_path = 'data/%d.csv' % file_idx
while os.path.isfile(img_path) and os.path.isfile(csv_path):
    print('Converting: ' + img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    X_Data = np.append(X_Data,x,axis=0)

    # features = model.predict(x)

    # output_str = ""

    has_gold = False
    with open(csv_path,'rb') as csv_data_file:
        reader  = csv.reader(csv_data_file)
        for row in reader:
            if 'gold' in row:
                has_gold = True

    Y_Data = np.append(Y_Data,[1 if has_gold else 0],axis=0)

    print(X_Data.shape)

    if (file_idx % 100) == 99:
        print('Saving')
        np.savez('all_images_%s.npz' % file_idx,X_Data,Y_Data)
        X_Data = np.empty([0,224,224,3])
        Y_Data = np.empty([0])

    file_idx = file_idx + 1
    img_path = 'data/%d.tif' % file_idx
    csv_path = 'data/%d.csv' % file_idx

