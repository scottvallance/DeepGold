# Turns a processed data directory in a LibSVM multi-label set
# Expects to run in the top level directory
# Assumes:
#   data/*.tif 														False colour images
#   data/*.csv 														Labels
#   mines_and_mineral_occurrences_all_shp/canonical_index.csv 		The label to index map

from __future__ import print_function
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import preprocess_input
from keras.models import Model
import numpy as np
import os.path
import csv

# Load canonical label dictionary
canonical = dict()
with open('mines_and_mineral_occurrences_all_shp/canonical_index.csv', 'rb') as csvfile:
	reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
	for row in reader:
		canonical[row['Label']] = row['Index']

# Load ResNet50

base_model = ResNet50(weights='imagenet')
model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)

output_file = open('data.svm','w')
file_idx = 0
img_path = 'data/%d.tif' % file_idx
csv_path = 'data/%d.csv' % file_idx
while os.path.isfile(img_path) and os.path.isfile(csv_path):
	print("Converting: " + img_path)

	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	features = model.predict(x)

	output_str = ""

	with open(csv_path,'rb') as csv_data_file:
		reader  = csv.reader(csv_data_file)
		for row in reader:
			output_str = ",".join(map(lambda x: canonical[x],row))

	for feature_idx, feature in enumerate(features[0,0,0,:]):
		if feature != 0:
			output_str = output_str + " " + str(feature_idx+1) + ":" + str(feature)

	print(output_str,file=output_file)

	file_idx = file_idx + 1
	img_path = 'data/%d.tif' % file_idx
	csv_path = 'data/%d.csv' % file_idx

output_file.close()