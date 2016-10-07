package myclassifier;

import java.util.*;
import weka.classifiers.*;
import weka.core.*;
import weka.core.Capabilities.Capability;

/**
 *
 * @author Visat
 */
public class MyJ48 extends Classifier {
    
    private MyJ48[] successors;
    private Attribute splitAttribute;    
    private double classValue;
    private double[] classDistribution;    
    private Attribute classAttribute;
    private double attributeThreshold;
    public MyJ48 head, parent;
    private boolean pruning = false;    
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capability.BINARY_CLASS);
        capabilities.enable(Capability.NOMINAL_CLASS);
        capabilities.enable(Capability.MISSING_CLASS_VALUES);
        capabilities.enable(Capability.MISSING_VALUES);

        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {        
        // Mengecek dapatkah classifier menghandle data
        getCapabilities().testWithFail(instances);
        
        // Penanganan missing value
        for (int j = 0; j < instances.numAttributes(); j++) {
            Attribute attr = instances.attribute(j);
            for (int k = 0; k < instances.numInstances(); k++) {
                Instance instance = instances.instance(k);
                if (instance.isMissing(attr)) {
                    instance.setValue(attr, fillMissingValue(instances, attr));                    
                }
            }
        }        
        // Menghapus instance dengan missing class
        instances.deleteWithMissingClass();
        buildTree(instances);
    }

    @Override
    public double classifyInstance(Instance instance) {
        if (splitAttribute == null)
            return classValue;        
        if (splitAttribute.isNominal())
            return successors[(int) instance.value(splitAttribute)].classifyInstance(instance);
        if (splitAttribute.isNumeric()) {            
            if (instance.value(splitAttribute) < attributeThreshold)
                return successors[0].classifyInstance(instance);
            else
                return successors[1].classifyInstance(instance);
        }
        else return -1;
    }
    
    @Override
    public String toString() {
        if ((classDistribution == null) && (successors == null)) {
            return "classifier.MyJ48: No model";
        }
        return "classifier.MyJ48\n\n" + treeToString(0);
    }    
    
    protected String treeToString(int level) {
        StringBuilder text = new StringBuilder();

        if (splitAttribute == null) {
            if (Instance.isMissingValue(classValue)) {
                text.append(": null");
            } else {
                text.append(": ").append(classAttribute.value((int) classValue));
            }
        } else {
            for (int j = 0; j < splitAttribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++) {
                    text.append("|  ");
                }
                text.append(splitAttribute.name()).append(" = ").append(splitAttribute.value(j));
                text.append(successors[j].treeToString(level + 1));
            }
        }
        return text.toString();
    }
    
    public MyJ48() {
        head = this;
    }

    public MyJ48(MyJ48 head, MyJ48 parent) {
        this.head = head;
        this.parent = parent;
    }

    public void setPruning(boolean enable) {
        pruning = enable;
    }

    private void prune(Instances instances) throws Exception {
        if (successors == null)
            return;        
        for (MyJ48 successor: successors) {
            successor.prune(instances);
            if (parent != null && pruneError(instances))
                break;                                
        }
    }

    private boolean pruneError(Instances instances) throws Exception {        
        double before = Util.percentageSplitRate(instances, head);        
        Attribute temp = this.parent.splitAttribute;
        this.parent.splitAttribute = null;
        double maxAfter = -Double.MAX_VALUE;
        double maxClass = -1;
        for (int x = 0; x < instances.numClasses(); x++) {
            this.parent.classValue = (double) x;
            double after = Util.percentageSplitRate(instances, head);
            if (after > maxAfter) {
                maxClass = x;
                maxAfter = after;
            }
        }

        this.parent.classValue = maxClass;        
        if (before >= maxAfter) {
            this.parent.splitAttribute = temp;
            return false;
        }
        return true;        
    }

    private double fillMissingValue(Instances instances, Attribute attribute) {
        int[] sum = new int[attribute.numValues()];
        for (int i = 0; i < instances.numInstances(); i++)
            sum[(int) instances.instance(i).value(attribute)]++;        
        return sum[Util.indexOfMax(sum)];
    }    
   
    public void buildTree(Instances data) throws Exception {
        if (data.numInstances() == 0)
            return;      
        
        if (pruning)
            prune(data);

        double[] IG = new double[data.numAttributes()];
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute att = data.attribute(i);
            if (data.classIndex() != att.index()) {
                if (att.isNominal()) {
                    IG[att.index()] = Util.calculateIG(data, att);
                } else {
                    IG[att.index()] = Util.calculateIGCont(data, att, bestAttributeCont(data, att));
                }
            }
        }

        splitAttribute = data.attribute(Util.indexOfMax(IG));
        if (splitAttribute.isNumeric())
            attributeThreshold = bestAttributeCont(data, splitAttribute);                    

        if (Util.equalValue(IG[splitAttribute.index()], 0)) {
            splitAttribute = null;
            classDistribution = new double[data.numClasses()];
            for (int i = 0; i < data.numInstances(); i++) {
                int inst = (int) data.instance(i).value(data.classAttribute());
                classDistribution[inst]++;
            }
            Util.normalizeClassDistribution(classDistribution);
            classValue = Util.indexOfMax(classDistribution);
            classAttribute = data.classAttribute();
        }
        else {
            Instances[] splitData;
            if (splitAttribute.isNominal()) {
                splitData = Util.splitData(data, splitAttribute);
            } else {
                splitData = Util.splitDataCont(data, splitAttribute, attributeThreshold);
            }

            if (splitAttribute.isNominal()) {
                successors = new MyJ48[splitAttribute.numValues()];                
                for (int j = 0; j < splitAttribute.numValues(); j++) {
                    successors[j] = new MyJ48(head, this);
                    successors[j].buildClassifier(splitData[j]);
                }
            } else {                
                successors = new MyJ48[2];                
                for (int j = 0; j < 2; j++) {
                    successors[j] = new MyJ48(head, this);
                    successors[j].buildClassifier(splitData[j]);
                }
            }
        }
    }

    public double bestAttributeCont(Instances instances, Attribute attribute) {
        instances.sort(attribute);
        Enumeration enumeration = instances.enumerateInstances();                
        double lastClass = instances.instance(0).classValue();
        double maxIG = -Double.MAX_VALUE;
        double maxThreshold = -Double.MAX_VALUE;        
        double lastValue = instances.instance(0).value(attribute);
        while (enumeration.hasMoreElements()) {
            Instance instance = (Instance) enumeration.nextElement();
            if (lastClass != instance.classValue()) {
                lastClass = instance.classValue();
                double a = lastValue;
                lastValue = instance.value(attribute);
                double threshold = a + ((lastValue - a) / 2);
                double curIG = Util.calculateIGCont(instances, attribute, threshold);
                if (maxIG < curIG) {
                    maxThreshold = threshold;
                    maxIG = curIG;
                }
            }
        }
        return maxThreshold;
    }
}
