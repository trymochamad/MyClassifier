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
    private boolean pruning;    
    private double attributeThreshold;

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
        for (int i = 0; i < instances.numAttributes(); ++i) {
            Attribute attribute = instances.attribute(i);
            for (int j = 0; j < instances.numInstances(); ++j) {
                Instance instance = instances.instance(j);
                if (instance.isMissing(attribute))
                    instance.setValue(attribute, fillMissingValue(instances, attribute));
            }
        }
        instances.deleteWithMissingClass();
        buildTree(instances);
    }
    
    @Override
    public double classifyInstance(Instance instance) {        
        if (splitAttribute == null)
            return classValue;        
        else if (splitAttribute.isNominal())
            return successors[(int) instance.value(splitAttribute)].classifyInstance(instance);                
        else if (splitAttribute.isNumeric()) {
            if (instance.value(splitAttribute) < attributeThreshold)
                return successors[0].classifyInstance(instance);
            else
                return successors[1].classifyInstance(instance);
        }
        else return -1;        
    }    
        
    private double fillMissingValue(Instances instances, Attribute attribute) {
        int[] sum = new int[attribute.numValues()];
        for (int i = 0; i < instances.numInstances(); ++i)
            sum[(int)instances.instance(i).value(attribute)]++;
        return sum[Util.indexOfMax(sum)];
    }
    
    public void setPruning(boolean enable){
        pruning = enable;
    }

    public void buildTree(Instances instances) throws Exception {        
        if (pruning)
            instances = prune(instances);
        
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
            }
            else {
                Instances[] splitData = Util.splitData(instances, splitAttribute);
                successors = new MyJ48[splitAttribute.numValues()];
                for (int i = 0; i < splitAttribute.numValues(); i++) {
                    successors[i] = new MyJ48();
                    successors[i].buildTree(splitData[i]);
                }
            }
        }
    }

    protected Instances prune(Instances instances) throws Exception {
        ArrayList<Integer> unsignificantAttributes = new ArrayList<>();
        Enumeration attEnum = instances.enumerateAttributes();
        while (attEnum.hasMoreElements()) {            
            Attribute att = (Attribute) attEnum.nextElement();
            double currentGainRatio = Util.calculateGainRatio(instances, att);
            if (currentGainRatio < 1.0) {
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
