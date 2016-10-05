from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model, load_model
import numpy as np
import os.path
import csv

model = load_model('../custom_resnet.h5')

def generate_data():
    while 1:
        X_Data = np.empty([0,224,224,3])
        Y_Data = np.empty([0])
        file_idx = 2200
        img_path = '../data/%d.tif' % file_idx
        csv_path = '../data/%d.csv' % file_idx
        while os.path.isfile(img_path) and os.path.isfile(csv_path):
            print img_path + " " + csv_path
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            X_Data = np.append(X_Data,x,axis=0)
            print img_path
            # if (file_idx % 20) == 19:
            yield(X_Data)
            X_Data = np.empty([0,224,224,3])
            file_idx = file_idx + 1
            img_path = '../data/%d.tif' % file_idx
            csv_path = '../data/%d.csv' % file_idx

predictions = model.predict_generator(generate_data(), 1)
