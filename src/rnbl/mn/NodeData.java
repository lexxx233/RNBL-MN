package rnbl.mn;

import java.util.HashMap;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instance;
import weka.core.Instances;

public class NodeData {
   double path = Double.NaN;
   NaiveBayesMultinomial nbm;
   Instances data;
   double cll; 
   
   public NodeData(Instances D, double fromP) throws Exception{
       path = fromP;
       nbm = new NaiveBayesMultinomial();
       data = D;
       nbm.buildClassifier(data);
       cll();
    }
   
   public HashMap<Double, Instances> splitD() throws Exception{
       HashMap<Double, Instances> sData = new HashMap<>();
       for(int j = 0; j < data.numInstances(); j++){
           double temp = nbm.classifyInstance(data.instance(j));
           if(sData.containsKey(temp)){
               Instances t = sData.get(temp);
               t.add(data.instance(j));
               sData.put(temp, t);
           }else{
               Instances t = new Instances(data);
               t.delete();
               t.add(data.instance(j));
               sData.put(temp, t);
           }
       }
       return sData;
   }
   
   private void cll() throws Exception{     
       for(int i = 0; i < data.numInstances(); i++){
           Instance ins = data.instance(i);
           double[] x = nbm.distributionForInstance(ins);
           for(int j = 0; j < data.numClasses(); j++){
                cll += Math.log(x[j]);
           }
       }
        cll *= data.numInstances();
    } 
}
