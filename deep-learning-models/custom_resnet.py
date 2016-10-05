from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from resnet50 import ResNet50
from keras.preprocessing import image
import numpy as np
import os.path
import csv

def generate_data():
    while 1:
        X_Data = np.empty([0,224,224,3])
        Y_Data = np.empty([0])
        file_idx = 0
        img_path = 'data/%d.tif' % file_idx
        csv_path = 'data/%d.csv' % file_idx
        while os.path.isfile(img_path) and os.path.isfile(csv_path):
            print img_path + " " + csv_path
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            X_Data = np.append(X_Data,x,axis=0)

            has_gold = False
            with open(csv_path,'rb') as csv_data_file:
                reader  = csv.reader(csv_data_file)
                for row in reader:
                    if 'gold' in row:
                        has_gold = True

            Y_Data = np.append(Y_Data,[1 if has_gold else 0],axis=0)

            print img_path + ", " + str(x.shape) + ": " + str([1 if has_gold else 0])

            if (file_idx % 20) == 19:
                yield(X_Data,Y_Data)
                X_Data = np.empty([0,224,224,3])
                Y_Data = np.empty([0])

            file_idx = file_idx + 1
            img_path = 'data/%d.tif' % file_idx
            csv_path = 'data/%d.csv' % file_idx


def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')
    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

input_shape = (224, 224, 3)
img_input = Input(shape=input_shape)
base_model = ResNet50(weights='imagenet',input_tensor=img_input)
pop_layer(base_model)
x = Dense(1, activation='softmax', name='fc1')(base_model.layers[-1].output)
print "New model defined"
model = Model(img_input, x)
print "New model compiled"
model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['accuracy'])
print "Fit generator"
model.fit_generator(generate_data(),samples_per_epoch=20,nb_epoch=100,verbose=2)
model.save('custom_resnet.h5')

