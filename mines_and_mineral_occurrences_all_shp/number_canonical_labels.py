# Number canonical labels

import csv

with open('label_map.csv', 'rb') as csvfile:
	canonical = dict()
	reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
	for row in reader:
		canonical[row['Canonical']] = True
	print "Label,Index"
	for index,label in enumerate(canonical):
		print '"'+label+'",'+str(index+1)
