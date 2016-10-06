package myclassifier;

import java.util.*;
import weka.classifiers.Classifier;
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
    private boolean prune;    

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
        getCapabilities().testWithFail(instances);        
        instances.deleteWithMissingClass();
        Instances nominalInstances = Util.toNominal(instances);
        buildTree(nominalInstances);
    }
    
    @Override
    public double classifyInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) throw new NoSupportForMissingValuesException("classifier.MyID3: This classifier can not handle missing value");
        if (splitAttribute == null) return classValue;
        else return successors[(int) instance.value(splitAttribute)].classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance)
            throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue()) throw new NoSupportForMissingValuesException("classifier.MyID3: Cannot handle missing values");
        if (splitAttribute == null) return classDistribution;
        else {
            if(splitAttribute.value(0).contains("<=")){
                int threshold = Integer.valueOf(splitAttribute.value(0).substring(2, 3));
                if (instance.value(splitAttribute) > threshold)
                    return successors[1].distributionForInstance(instance);
                else
                    return successors[0].distributionForInstance(instance);
            }
            return successors[(int) instance.value(splitAttribute)].distributionForInstance(instance);
        }
    }    
    
    @Override
    public String toString() {
        if ((classDistribution == null) && (successors == null)) {
            return "classifier.MyID3: No model";
        }
        return "classifier.MyID3\n\n" + treeToString(0);
    }

    protected String treeToString(int level) {
        StringBuilder text = new StringBuilder();

        if (splitAttribute == null) {
            if (Instance.isMissingValue(classValue))
                text.append(": null");
            else
                text.append(": ").append(classAttribute.value((int) classValue));
        } else {
            for (int j = 0; j < splitAttribute.numValues(); j++) {
                text.append("\n");
                for (int i = 0; i < level; i++)
                    text.append("|  ");
                text.append(splitAttribute.name()).append(" = ").append(splitAttribute.value(j));
                text.append(successors[j].treeToString(level + 1));
            }
        }
        return text.toString();
    }
    
    public void enablePrune(boolean enable){
        prune = enable;
    }

    public void buildTree(Instances instances) throws Exception {        
        if (prune)
            instances = doPruning(instances);
        
        if (instances.numInstances() == 0) {
            splitAttribute = null;
            classValue = Instance.missingValue();
            classDistribution = new double[instances.numClasses()];
        }
        else {            
            double[] gainRatio = new double[instances.numAttributes()];
            Enumeration attEnum = instances.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                gainRatio[att.index()] = Util.calculateGainRatio(instances, att);
            }
            splitAttribute = instances.attribute(Util.indexOfMax(gainRatio));
            
            if (Util.equalValue(gainRatio[splitAttribute.index()], 0)) {
                splitAttribute = null;
                classDistribution = new double[instances.numClasses()];
                for (int i = 0; i < instances.numInstances(); i++) {
                    Instance inst = (Instance) instances.instance(i);
                    classDistribution[(int) inst.classValue()]++;
                }
                Util.normalizeClassDistribution(classDistribution);
                classValue = Util.indexOfMax(classDistribution);
                classAttribute = instances.classAttribute();
            }
            else {
                Instances[] splitData = Util.splitDataBasedOnAttribute(instances, splitAttribute);
                successors = new MyJ48[splitAttribute.numValues()];
                for (int i = 0; i < splitAttribute.numValues(); i++) {
                    successors[i] = new MyJ48();
                    successors[i].buildTree(splitData[i]);
                }
            }
        }
    }

    protected Instances doPruning(Instances instances) throws Exception {
        ArrayList<Integer> unsignificantAttributes = new ArrayList();
        Enumeration attEnum = instances.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            double currentGainRatio;
            Attribute att = (Attribute) attEnum.nextElement();
            currentGainRatio = Util.calculateGainRatio(instances, att);
            if (currentGainRatio < 1) {
                unsignificantAttributes.add(att.index() + 1);
            }
        }
        if (unsignificantAttributes.size() > 0) {
            StringBuilder unsignificant = new StringBuilder();
            int i = 0;
            for (Integer current : unsignificantAttributes) {
                unsignificant.append(current.toString());
                if (i != unsignificantAttributes.size()-1) {
                    unsignificant.append(",");
                }
                i++;
            }
            return wekaCode.removeAttributes(instances, unsignificant.toString());
        }
        else return instances;        
    }            
}
