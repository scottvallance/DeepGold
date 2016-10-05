# Turn multi-label data into binary data

import sys, os.path
from sys import argv
from os import system
from string import *

if len(argv) < 4:
	print "Usage: %s training_file out_file label" % (argv[0])
	sys.exit(1)

assert os.path.exists(argv[1]),"training_file not found."

out_file = open(argv[2],"w")
in_file = open(argv[1],"r")
for line in in_file:
	spline = split(line)

	labels = []
	if spline[0].find(':') == -1:
		labels = split(spline[0],',')
		labels.sort()

	if argv[3] in labels:
		out_file.write("1 %s\n"%(join(spline[1:])))
	else:
		out_file.write("0 %s\n"%(join(spline[1:])))

out_file.close()
in_file.close()

