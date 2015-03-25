# RNBL-MN
Recursive Naive Bayes Multinomial 

Implementation of RNBL-MN 

Built on top of WEKA. References can be found [here](http://link.springer.com/chapter/10.1007%2F11731139_8#page-1)

## Usage
Load in your training and test (ARFF format)
```java
import java.io.File;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

...

ArffLoader loader = new ArffLoader();

loader.setFile(new File("train.arff"));
Instances trainSet = loader.getDataSet();
trainSet.setClassIndex(dataSet.numAttributes()-1);

loader.setFile(new File("test.arff"));
Instances testSet = loader.getDataSet();
testSet.setClassIndex(dataSet.numAttributes()-1);
```
Train RNBL
```java
RNBLMN classifier = new RNBLMN(trainSet, Double.NaN);
classifier.build(); //Recursively construct NB Tree
```
Classify the an instance in the test set (i.e. first instance)
```java
classifier.classify(testSet.firstInstance()); 
//This will return a double - the classValue which the instance is classified into
```
Or evaluate performance on the entire test set
```java
classifier.eval(testSet);
```
