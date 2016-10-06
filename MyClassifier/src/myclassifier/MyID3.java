/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

import java.util.*;

import static weka.core.Utils.log2;

/**
 *
 * @author Scemo
 */
public class MyID3 extends Classifier {
    private MyID3[] successors;
    private Attribute splitAttribute;
    private double classValue;
    private double[] classDistribution;
    private Attribute classAttribute;
    
    @Override
    public Capabilities getCapabilities() {
        Capabilities capabilities = super.getCapabilities();
        capabilities.disableAll();

        capabilities.enable(Capability.NOMINAL_ATTRIBUTES);
        capabilities.enable(Capability.NUMERIC_ATTRIBUTES);
        capabilities.enable(Capability.BINARY_CLASS);
        capabilities.enable(Capability.NOMINAL_CLASS);
        capabilities.enable(Capability.MISSING_CLASS_VALUES);

        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Mengecek dapatkah classifier menghandle data
        getCapabilities().testWithFail(instances);

        // menghapus instance dengan missing class
        instances.deleteWithMissingClass();
        Instances nominalInstances = toNominal(instances);
        buildTree(nominalInstances);
    }
    
    public void buildTree(Instances instances) throws Exception {
        // Mengecek ada tidaknya instance yang mencapai node ini
        if (instances.numInstances() == 0) {
            splitAttribute = null;
            classValue = Instance.missingValue();
            classDistribution = new double[instances.numClasses()];
            return;
        } else {
            // Mencari Information Gain maksimum dari atribut
            double[] infoGains = new double[instances.numAttributes()];
            Enumeration attEnum = instances.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute attr = (Attribute) attEnum.nextElement();
                infoGains[attr.index()] = calculateIG(instances, attr);
            }
            splitAttribute = instances.attribute(indexWithMaxValue(infoGains));

            // Jika IG max = 0, buat daun dengan label kelas mayoritas
            // Jika tidak, buat successor
            if (equalValue(infoGains[splitAttribute.index()], 0)) {
                splitAttribute = null;
                classDistribution = new double[instances.numClasses()];

                for (int i = 0; i < instances.numInstances(); i++) {
                    Instance inst = (Instance) instances.instance(i);
                    classDistribution[(int) inst.classValue()]++;
                }

                normalizeClassDistribution(classDistribution);
                classValue = indexWithMaxValue(classDistribution);
                classAttribute = instances.classAttribute();
            } else {
                Instances[] splitData = splitDataBasedOnAttribute(instances, splitAttribute);
                successors = new MyID3[splitAttribute.numValues()];
                for (int j = 0; j < splitAttribute.numValues(); j++) {
                    successors[j] = new MyID3();
                    successors[j].buildTree(splitData[j]);
                }
            }
        }
    }
    
    public Instances toNominal(Instances data) throws Exception {
        for (int n=0; n<data.numAttributes(); n++) {
            Attribute att = data.attribute(n);
            if (data.attribute(n).isNumeric()) {
                HashSet<Integer> uniqueValues = new HashSet();
                for (int i = 0; i < data.numInstances(); ++i)
                    uniqueValues.add((int) (data.instance(i).value(att)));

                List<Integer> dataValues = new ArrayList<Integer>(uniqueValues);
                dataValues.sort(new Comparator<Integer>() {
                    public int compare(Integer o1, Integer o2) {
                        if(o1 > o2){
                            return 1;
                        }else{
                            return -1;
                        }
                    }
                });

                // Search for threshold and get new Instances
                double[] infoGains = new double[dataValues.size() - 1];
                Instances[] tempInstances = new Instances[dataValues.size() - 1];
                for (int i = 0; i < dataValues.size() - 1; ++i) {
                    tempInstances[i] = setAttributeThreshold(data, att, dataValues.get(i));
                    infoGains[i] = computeIG(tempInstances[i], tempInstances[i].attribute(att.name()));
                }
                data = new Instances(tempInstances[indexWithMaxValue(infoGains)]);
            }
        }
        return data;
    }
    
    private static Instances setAttributeThreshold(Instances data, Attribute attr, int threshold) throws Exception {
        Instances temp = new Instances(data);
        // Add thresholded attribute
        Add filter = new Add();
        filter.setAttributeName("thresholded " + attr.name());
        filter.setAttributeIndex(String.valueOf(attr.index() + 2));
        filter.setNominalLabels("<=" + threshold + ",>" + threshold);
        filter.setInputFormat(temp);
        Instances thresholdedData = Filter.useFilter(data, filter);

        for (int i=0; i<thresholdedData.numInstances(); i++) {
            if ((int) thresholdedData.instance(i).value(thresholdedData.attribute(attr.name())) <= threshold)
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + attr.name()), "<=" + threshold);
            else
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + attr.name()), ">" + threshold);
        }
        thresholdedData = wekaCode.removeAttributes(thresholdedData, String.valueOf(attr.index() + 1));
        thresholdedData.renameAttribute(thresholdedData.attribute("thresholded " + attr.name()), attr.name());
        return thresholdedData;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        System.out.println(instance);
        if (instance.hasMissingValue())
            throw new NoSupportForMissingValuesException("classifier.MyID3: This classifier can not handle missing value");
        if (splitAttribute == null)
            return classValue;
        else
            return successors[(int) instance.value(splitAttribute)].classifyInstance(instance);
    }
}
