#########
# MultilayerPerceptronClassifier

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import MultilayerPerceptronClassifier

data = spark.read.format("libsvm").load("gold.svm")

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

resampleGold = trainingData.filter(trainingData['label'] == 1.0)
nonGold = trainingData.filter(trainingData['label']== 0.0)
trainingData = resampleGold.sample(withReplacement=True,fraction=float(nonGold.count())/resampleGold.count(),seed=11L).union(nonGold)

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
# Automatically identify categorical features, and index them.
# Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)


layers = [2048, 512, 2]
# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=1000, layers=layers, blockSize=128, seed=1234)
# train the model
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, trainer])
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(trainingData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Training Accuracy = %g" % (accuracy))

# compute accuracy on the test set
predictions = model.transform(testData)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Accuracy = %g" % (accuracy))

total_gold = predictions.filter(predictions['label'] == 1)
accurate_gold = total_gold.filter(predictions['prediction'] == 1)
print("Gold Recall: " + str(float(accurate_gold.count())/total_gold.count()))

total_pred_gold = predictions.filter(predictions['prediction'] == 1)
print("Gold Precision: "+ str(float(accurate_gold.count())/total_pred_gold.count()))

total_ngold =  predictions.filter(predictions['label'] == 0)
accurate_ngold = total_ngold.filter(predictions['prediction'] == 0)
print("Non-Gold Recall: " + str(float(accurate_ngold.count())/total_ngold.count()))

total_pred_ngold = predictions.filter(predictions['prediction'] == 0)
print("Non-Gold Precision: "+ str(float(accurate_ngold.count())/total_pred_ngold.count()))


Po = accuracy
Pe = (predictions.filter(predictions['prediction'] == 1).count()*predictions.filter(predictions['label'] == 1).count() +\
	predictions.filter(predictions['prediction'] == 0).count()*predictions.filter(predictions['label'] == 0).count())/(float(testData.count())**2)
goldKappa = 1 - (1 - Po)/(1 - Pe)
print("Gold Kappa = " + str(goldKappa))

print(str(accurate_gold.count()) + " " + str(total_gold.count()-accurate_gold.count()))
print(str(total_ngold.count()-accurate_ngold.count()) + " " + str(accurate_ngold.count()))