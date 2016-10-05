# Processes minerals.csv to find unique labels, these are then hand matched to canonical names, ie 'au' to 'gold'

import csv

with open('minerals.csv', 'rb') as csvfile:
	minerals = dict()
	reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
	for row in reader:
		for mineral in map(lambda x: x.strip(),row['DEPOSIT_CO'].lower().split(',')):
			if mineral in minerals:
				minerals[mineral].append(reader.line_num)
			else:
				minerals[mineral] = [reader.line_num]
	for mineral in minerals:
		print mineral + ","  + ",".join(map(str,minerals[mineral]))
