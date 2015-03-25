package rnbl.mn;

import java.io.File;
import java.util.Random;
import weka.core.Instances;
import weka.core.converters.ArffLoader;

/**
 * Ain't nothing in here but the main thread
 * @author txl252
 */
public class Main {
    public static void main(String[] args) throws Exception {
        if(args.length != 2){
            System.err.println("usage java -jar RNBL-MN.jar <data.arff> <CV-Folds>");
            System.exit(1);
        }
        ArffLoader loader = new ArffLoader();
        loader.setFile(new File(args[0]));
        int folds = Integer.parseInt(args[1]);
        
        Instances structure = loader.getStructure();
        structure.setClassIndex(structure.numAttributes()-1);
        Instances dataSet = loader.getDataSet();
        dataSet.setClassIndex(dataSet.numAttributes()-1);
        
        Instances randData = new Instances(dataSet);
        randData.randomize(new Random());
        randData.stratify(folds);
        
        double[] cm = {0, 0, 0, 0};
        for(int n = 0; n < folds; n++){
            Instances train = randData.trainCV(folds, n);
            Instances test = randData.testCV(folds, n);
            
            RNBLMN classifier = new RNBLMN(train, Double.NaN);
            classifier.build();
            System.out.println("Depth of NB tree = " + Integer.toString(classifier.root.getDepth()));
            double[] eval = classifier.eval(test);
            for(int i = 0; i < cm.length; i++){
                cm[i] += eval[i];
            }
        }
        System.out.println("Accuracy : "  + Double.toString((cm[0]+cm[1])/(cm[0]+cm[1]+cm[2]+cm[3])));
        System.out.println("Precision : " + Double.toString(cm[0]/(cm[0]+cm[2])));
        double rec = cm[0]/(cm[0]+cm[3]);
        double pre = cm[0]/(cm[0]+cm[2]);
        System.out.println("Recall : " + Double.toString(rec));
        System.out.println("F measure : " + Double.toString(2 * pre * rec / (pre + rec)));
    }
}
