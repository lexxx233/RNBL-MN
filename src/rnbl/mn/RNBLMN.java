package rnbl.mn;

import java.util.Collections;
import java.util.Enumeration;
import java.util.HashMap;
import javax.swing.tree.DefaultMutableTreeNode;

import weka.core.Instance;
import weka.core.Instances;

public class RNBLMN {
    
    DefaultMutableTreeNode root;
    Instances train;
    
    public RNBLMN(Instances _train, double fromP) throws Exception{
        root = new DefaultMutableTreeNode();
        train = _train;
        root.setUserObject(new NodeData(train, fromP));
    }
    
    public double cdml(DefaultMutableTreeNode n){
        double cdml = 0.0;
        Enumeration a = n.depthFirstEnumeration();
        while(a.hasMoreElements()){
            cdml += ((NodeData) ((DefaultMutableTreeNode) a.nextElement()).getUserObject()).cll; //SUM(CLL)
        }
        return cdml - (Math.log(train.numInstances()/2) * Collections.list(n.depthFirstEnumeration()).size());
    }
    
    public void build() throws Exception{
        root = buildClassifier(root);
    }
    
    public DefaultMutableTreeNode buildClassifier(DefaultMutableTreeNode n) throws Exception{
         HashMap<Double, Instances> cur = ((NodeData) n.getUserObject()).splitD();
         double currentCDML = cdml(n);
         for(double i : cur.keySet()){
             RNBLMN temp = new RNBLMN(cur.get(i), i);
             n.add(temp.root);
         }
         double newCDML = cdml(n) - currentCDML;
         if(newCDML <= currentCDML){ //Reject changes when CDML doesn't improve
             n.removeAllChildren();
         }else{ 
             n.removeAllChildren();
             for(double i : cur.keySet()){
                RNBLMN temp = new RNBLMN(cur.get(i), i);
                temp.root = buildClassifier(temp.root);
                n.add(temp.root);
            }
         }
         return n;
    }
    
    public double classify(Instance ins) throws Exception{
        DefaultMutableTreeNode node = root;
        
        while(node.getDepth()!= 0){
            double i = ((NodeData) node.getUserObject()).nbm.classifyInstance(ins);
            Enumeration temp = node.children();
            while(temp.hasMoreElements()){
                DefaultMutableTreeNode dn = (DefaultMutableTreeNode) temp.nextElement();
                if(((NodeData) dn.getUserObject()).path == i){
                    node = dn;
                    break;
                }
            }
        }
        
        return ((NodeData) node.getUserObject()).nbm.classifyInstance(ins);
    }
    
    public double[] eval(Instances testdata) throws Exception{
        double[] res = {0.0, 0.0, 0.0, 0.0}; //TP TN FP FN
        for(int i = 0; i < testdata.numInstances(); i++){
            Instance ins = testdata.instance(i);
            if(classify(ins) == ins.classValue()){
                if(ins.classValue() == 0.0){
                    res[1] += 1;
                }else{
                    res[0] += 1;
                }
            }else{
                if(ins.classValue() == 0.0){
                    res[2] += 1;
                }else{
                    res[3] += 1;
                }
            }
        }
        
        return res;
    }
         
}
