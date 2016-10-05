#########
# Logistic, SVM, NaiveBayes

from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel

data = MLUtils.loadLibSVMFile(sc, 'gold.svm')

splits = data.randomSplit([0.7, 0.3], seed = 11L)
training = splits[0].cache()
test = splits[1]

resampleGold = training.filter(lambda x: x.label == 1.0)
nonGold = training.filter(lambda x: x.label == 0.0)
training = resampleGold.sample(withReplacement=True,fraction=float(nonGold.count())/resampleGold.count(),seed=11L).union(nonGold)

numIterations = 1000

# model = LogisticRegressionWithLBFGS.train(training, iterations=numIterations)
# model = SVMWithSGD.train(training, iterations=numIterations)
model = NaiveBayes.train(training)

labelsAndPredsTrain = training.map(lambda p: (p.label, model.predict(p.features)))
trainErr = labelsAndPredsTrain.filter(lambda (v, p): v != p).count() / float(training.count())
print("Training Accuracy = " + str(1.0-trainErr))

labelsAndPredsTest = test.map(lambda p: (p.label, model.predict(p.features)))
testErr = labelsAndPredsTest.filter(lambda (v, p): v != p).count() / float(test.count())
print("Test Accuracy = " + str(1.0-testErr))

goldAcc = labelsAndPredsTest.filter(lambda (v,p): (v == 1) & (p == 1)).count() / float(labelsAndPredsTest.filter(lambda (v,p): v == 1).count())
print("Gold Recall = " + str(goldAcc))

goldPre = labelsAndPredsTest.filter(lambda (v,p): (v == 1) & (p == 1)).count() / float(labelsAndPredsTest.filter(lambda (v,p): p == 1).count())
print("Gold Precision = " + str(goldPre))

nongoldAcc = labelsAndPredsTest.filter(lambda (v,p): (v == 0) & (p == 0)).count() / float(labelsAndPredsTest.filter(lambda (v,p): v == 0).count())
print("Non-Gold Recall = " + str(nongoldAcc))

nongoldPre= labelsAndPredsTest.filter(lambda (v,p): (v == 0) & (p == 0)).count() / float(labelsAndPredsTest.filter(lambda (v,p): p == 0).count())
print("Non-Gold Precision = " + str(nongoldPre))


Po = 1.0-testErr
Pe = (labelsAndPredsTest.filter(lambda (v,p): (p == 1)).count()*labelsAndPredsTest.filter(lambda (v,p): (v == 1)).count() +\
	labelsAndPredsTest.filter(lambda (v,p): (p == 0)).count()*labelsAndPredsTest.filter(lambda (v,p): (v == 0)).count())/(float(test.count())**2)
goldKappa = 1 - (1 - Po)/(1 - Pe)
print("Gold Kappa = " + str(goldKappa))

print(str(labelsAndPredsTest.filter(lambda (v,p): (v == 1) & (p == 1)).count()) + " " + str(labelsAndPredsTest.filter(lambda (v,p): (v == 1) & (p == 0)).count()))
print(str(labelsAndPredsTest.filter(lambda (v,p): (v == 0) & (p == 1)).count()) + " " + str(labelsAndPredsTest.filter(lambda (v,p): (v == 0) & (p == 0)).count()))