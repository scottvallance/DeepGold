# Process minerals data from the mines_and_mineral_occurences_all_shp directory and match to images in the
# ASTERGeoscienceMap directory.
# Assumes:
#   ASTERGeoscienceMap/file_extents_trans.csv: 				A list of longitude and latitude for the image maps
#   mines_and_mineral_occurrences_all_shp/label_map.csv: 	A mapping of used mineral names to canonical names
#   mines_and_mineral_occurrences_all_shp/minerals.csv: 	A list of mineral finds with point data (LONGITUDE and LATITUDE) and 
#															DEPOSIT_CO a list of minerals
#   ASTERGeoscienceMap/<image>.tif 							A set of images matching the used features and map sections
#   data/													Where the output images and csv files are generated
#
# Uses the following images for the colour bands of the output:
#   R: _AlOH_Group_Content
#   G: _Kaolin_Group_Content
#   B: _MgOH_Group_Content
#
# Generates a script that relies on gdal_translate and gdal_merge and outputs to data
 
from __future__ import print_function
import csv
from shapely.geometry import Polygon, Point

section_extents = dict()
img_size_half = 112 # this is half of our target image size, although it could be larger and the images can

# Load the extents
with open('ASTERGeoscienceMap/file_extents_trans.csv') as extents_csv:
	extents_reader = csv.DictReader(extents_csv, delimiter=',', quotechar='"')
	for row in extents_reader:
		poly = Polygon(([float(row['Upper Left Lon']),float(row['Upper Left Lat'])],
  					[float(row['Upper Right Lon']),float(row['Upper Right Lat'])],
					[float(row['Lower Right Lon']),float(row['Lower Right Lat'])],
					[float(row['Lower Left Lon']),float(row['Lower Left Lat'])]))
		pixel_size_lan = Point((img_size_half*(poly.exterior.coords[1][0]-poly.exterior.coords[0][0])/float(row['width']),img_size_half*(poly.exterior.coords[1][1]-poly.exterior.coords[0][1])/float(row['width'])))
		pixel_size_lon = Point((img_size_half*(poly.exterior.coords[0][0]-poly.exterior.coords[3][0])/float(row['height']),img_size_half*(poly.exterior.coords[0][1]-poly.exterior.coords[3][1])/float(row['height'])))
		section_extents[row['Section']] = {'poly':poly,'pixel_size_lat':pixel_size_lan,'pixel_size_lon':pixel_size_lon}

# Load the label mapping
mineral_map = dict()
with open('mines_and_mineral_occurrences_all_shp/label_map.csv', 'rb') as labels_csv:
	reader = csv.DictReader(labels_csv, delimiter=',', quotechar='"')
	for row in reader:
		mineral_map[row['Original']] = row['Canonical']

# Load the point mineral data
mineral_info = []
with open('mines_and_mineral_occurrences_all_shp/minerals.csv', 'rb') as minerals_csv:
	reader = csv.DictReader(minerals_csv, delimiter=',', quotechar='"')
	for row in reader:
		pnt = Point((float(row['LONGITUDE']), float(row['LATITUDE'])))
		info = dict()
		info['pnt'] = pnt
		info['used'] = False
		info['minerals'] = dict()
		for mineral in map(lambda x: x.strip(),row['DEPOSIT_CO'].lower().split(',')):
			info['minerals'][mineral_map[mineral]] = True
		for section in section_extents:
			if section_extents[section]['poly'].contains(pnt):
				info['section'] = section
		mineral_info.append(info)

# Turn the point minerals in polygons that contain minerals from all points in that poly
total_poly = 0
polys_by_section = dict()
for info in mineral_info:
	if not info['used']:
		total_poly = total_poly + 1
		info['used'] = True
		section = info['section']
 		pixel_size_lat = section_extents[section]['pixel_size_lat']
		pixel_size_lon = section_extents[section]['pixel_size_lon']
		poly = dict()
		poly['poly'] = Polygon(([info['pnt'].coords[0][0]-pixel_size_lat.coords[0][0]+pixel_size_lon.coords[0][0],
							info['pnt'].coords[0][1]-pixel_size_lat.coords[0][1]+pixel_size_lon.coords[0][1]],
						[info['pnt'].coords[0][0]+pixel_size_lat.coords[0][0]+pixel_size_lon.coords[0][0],
							info['pnt'].coords[0][1]+pixel_size_lat.coords[0][1]+pixel_size_lon.coords[0][1]],
						[info['pnt'].coords[0][0]+pixel_size_lat.coords[0][0]-pixel_size_lon.coords[0][0],
							info['pnt'].coords[0][1]+pixel_size_lat.coords[0][1]-pixel_size_lon.coords[0][1]],
						[info['pnt'].coords[0][0]-pixel_size_lat.coords[0][0]-pixel_size_lon.coords[0][0],
							info['pnt'].coords[0][1]-pixel_size_lat.coords[0][1]-pixel_size_lon.coords[0][1]]))
		poly['minerals'] = info['minerals'].copy()
		cnt = 0
		for info2 in mineral_info:
			if poly['poly'].contains(info2['pnt']):
				info2['used'] = True
				poly['minerals'].update(info2['minerals'])
				cnt = cnt + 1	
		if not section in polys_by_section:
			polys_by_section[section] = [poly]
		else:
			polys_by_section[section].append(poly)

command_file = open('process_cmd.sh','w')
print('#!/bin/bash',file=command_file)

# Process each polygon and generate false colour image for each polygon as well as a csv list of minerals
bands = ['_AlOH_Group_Content','_Kaolin_Group_Index','_MgOH_Group_Content']
poly_index = 0
for section in polys_by_section:
	for poly in polys_by_section[section]:
		ul = [poly['poly'].exterior.coords[0][0],poly['poly'].exterior.coords[0][1]]
		lr = [poly['poly'].exterior.coords[2][0],poly['poly'].exterior.coords[2][1]]
		for band in bands:
			section_cmd = 'gdal_translate -projwin %f %f %f %f ASTERGeoscienceMap/%s%s.tif %s.tif' % (ul[0],ul[1],lr[0],lr[1],section,band,band)
			print(section_cmd,file=command_file)
		merge_cmd = 'gdal_merge.py -o data/%d.tif -separate %s.tif %s.tif %s.tif' %(poly_index,bands[0],bands[1],bands[2])
		print(merge_cmd,file=command_file)
		data_cmd = 'echo "' + ','.join(poly['minerals'].keys()) + '" > data/%d.csv' % poly_index
		print(data_cmd,file=command_file)
		poly_index = poly_index + 1

command_file.close()


