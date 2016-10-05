# Deep Gold

This repository explores using convolution networks to search for minerals. It relies on Keras, Spark (pyspark), GDAL and shapely.

### Acknowledgements
This project relies on data from [the ASTER project at Geoscience Australia](http://www.geoscience.gov.au/), the [SARIG from the State Government of South Australia](https://sarig.pir.sa.gov.au/Map) and [code from Keras](https://github.com/fchollet/deep-learning-models). Applicable licenses are preserved in their respective directories.

### Installation
First install [TensorFlow](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html), [Keras](https://keras.io/#installation), [Spark (for pyspark)](http://spark.apache.org/downloads.html), [GDAL](https://trac.osgeo.org/gdal/wiki/DownloadingGdalBinaries) (or from brew) and [Shapely](https://pypi.python.org/pypi/Shapely).

### How to run
First get the data by running the following:

`wget -i ASTERGeoscienceMap/files.txt -P ASTERGeoscienceMap`

Generate a bunch of gdal commands to slice up the large ASTERGeoscienceMaps and match with the mineral data by runing:

`python process_minerals.py`

Then run the resulting process_cmd.sh:

`chmod u+x process_cmd.sh; ./process_cmd.sh`

The resulting data directory should have about 3000 tif images and csv files. We can now process these with Keras to get features:

`python deep-learning-models/feature_run.py`

This creates 'data.svm' a LibSVM format data file with multiple labels (one for each mineral in the segment). For the purposes of using the single or binary label classifiers in Spark we then cut this down to show only gold or not gold labels by running:

`python convert_to_binary_svm.py data.svm gold.svm 3`

Where 3 is the label index for gold, which can be seen in the file:

`mines_and_mineral_occurences_all_shp/label_map.csv`

The file 'gold.svm' is ready for spark. The files can be run by opening pyspark and running execfile as follows:

```
> pyspark

Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 2.0.0
      /_/

Using Python version 2.7.10 (default, Jul 14 2015 19:46:27)
SparkSession available as 'spark'.
>>> execfile(<file>)
```

Where `<file>` is one of `'spark_logistic_svm.py'` or `'spark_forest.py'` or `'spark_mlpc.py'`

