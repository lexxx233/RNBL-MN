# RNBL-MN
Recursive Naive Bayes Multinomial 

Implementation of RNBL-MN 

Built on top of WEKA. References can be found [here](http://link.springer.com/chapter/10.1007%2F11731139_8#page-1)

## Usage
Load in your training and test (ARFF format)
```java
ArffLoader loader = new ArffLoader();

loader.setFile("train.arff");
Instances trainSet = loader.getDataSet();
trainSet.setClassIndex(dataSet.numAttributes()-1);

loader.setFile("test.arff")
Instances testSet = loader.getDataSet();
testSet.setClassIndex(dataSet.numAttributes()-1);
```
Train RNBL
```java
RNBLMN classifier = new RNBLMN(trainSet, Double.NaN);
classifier.build();
```
Classify the an instance in test set (i.e. first instance)
```java
classifier.classify(testSet.firstInstance()); 
//This will return a double - the classValue which the instance is classified into
```
