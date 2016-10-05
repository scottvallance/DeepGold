#!/bin/bash

files=*Kaolin*
for f in $files; do
	gdalinfo -noct -nomd -nogcp -norat -nofl $f | sed -n '/^Files/p;/^Upper/p;/^Lower/p;/^Size/p'
done