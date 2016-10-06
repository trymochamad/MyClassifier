/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

/**
 *
 * @author Visat
 */
public class Util {
    public static int indexOfMax(double[] array) {
        double max = -Double.MAX_VALUE;
        int idx = -1;
        
        for (int i = 0; i < array.length; ++i) {
            if (Double.compare(array[i], max) > 0) {
                max = array[i];
                idx = i;                
            }            
        }
        return idx;
    }
    
    public static int indexOfMax(int[] array) {
        double max = Integer.MIN_VALUE;
        int idx = -1;
        
        for (int i = 0; i < array.length; ++i) {
            if (array[i] > max) {
                max = array[i];
                idx = i;                
            }            
        }
        return idx;
    }
    
    public static boolean equalValue(double a, double b) {
        final double epsilon = 1e-6;
        return ((a == b) || Math.abs(a - b) < epsilon);
    }
    
    public static double log2(double val) {
        return equalValue(val, 0) ? 0.0 : (Math.log(val) / Math.log(2));
    }       
        
    public static void normalizeClassDistribution(double[] array) {
        double sum = 0;
        for (double d: array) sum += d;
        if (!Double.isNaN(sum) && sum != 0) {
            for (int i = 0; i < array.length; ++i) array[i] /= sum;
        }
    }
    
    public static double calculateE(Instances instances) {
        double[] labelCounts = new double[instances.numClasses()];
        for (int i = 0; i < instances.numInstances(); ++i)
            labelCounts[(int) instances.instance(i).classValue()]++;

        double entropy = 0.0;
        for (int i = 0; i < labelCounts.length; i++) {
            if (labelCounts[i] > 0) {
                double proportion = labelCounts[i] / instances.numInstances();
                entropy -= (proportion) * log2(proportion);
            }
        }
        return entropy;
    }   
    
    public static double calculateGainRatio(Instances data, Attribute attribute) {
        double IG = calculateIG(data, attribute);
        double IV = calculateIntrinsicValue(data, attribute);
        if (IG == 0 || IV == 0)
            return 0;
        return IG / IV;
    }

    private static double calculateIntrinsicValue(Instances data, Attribute attribute) {
        double IV = 0;
        Instances[] splitData = splitData(data, attribute);
        for (int i = 0; i < attribute.numValues(); i++){
            if (splitData[i].numInstances() > 0) {
                double proportion = (double)splitData[i].numInstances() / (double)data.numInstances();
                IV -= ( proportion * Util.log2(proportion));
            }
        }
        return IV;
    }  
    
    public static double calculateIG(Instances instances, Attribute attribute) {
        double IG = calculateE(instances);
        int missingCount = 0;
        Instances[] splitData = splitData(instances, attribute);
        for (int j = 0; j < attribute.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                IG -= ((double) splitData[j].numInstances() /
                        (double) instances.numInstances()) *
                        calculateE(splitData[j]);
            }
        }

        for (int i = 0; i < instances.numInstances(); i++){
            Instance instance = instances.instance(i);
            if (instance.isMissing(attribute))
                missingCount++;            
        }        
        return IG * (instances.numInstances() - missingCount / instances.numInstances());
    }
    
    public static Instances[] splitData(Instances instances, Attribute attribute) {
        Instances[] splittedData = new Instances[attribute.numValues()];

        for (int i = 0; i < attribute.numValues(); i++)
            splittedData[i] = new Instances(instances, instances.numInstances());

        for (int i = 0; i < instances.numInstances(); i++) {
            int attValue = (int) instances.instance(i).value(attribute);
            splittedData[attValue].add(instances.instance(i));
        }
//
//        for (Instances currentSplitData: splittedData)
//            currentSplitData.compactify();

        return splittedData;
    }
    
    public static Instances setAttributeThreshold(Instances data, Attribute att, int threshold) throws Exception {
        Instances temp = new Instances(data);        
        Add filter = new Add();
        filter.setAttributeName("thresholded " + att.name());
        filter.setAttributeIndex(String.valueOf(att.index() + 2));
        filter.setNominalLabels("<=" + threshold + ",>" + threshold);
        filter.setInputFormat(temp);

        Instances thresholdedData = Filter.useFilter(data, filter);

        for (int i=0; i < thresholdedData.numInstances(); i++) {
            if ((int) thresholdedData.instance(i).value(thresholdedData.attribute(att.name())) <= threshold)
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + att.name()), "<=" + threshold);
            else
                thresholdedData.instance(i).setValue(thresholdedData.attribute("thresholded " + att.name()), ">" + threshold);
        }
        thresholdedData = wekaCode.removeAttributes(thresholdedData, String.valueOf(att.index() + 1));
        thresholdedData.renameAttribute(thresholdedData.attribute("thresholded " + att.name()), att.name());
        return thresholdedData;
    }
    
    public static Instances toNominal(Instances data) throws Exception {
        for (int n = 0; n < data.numAttributes(); n++) {
            Attribute att = data.attribute(n);
            if (data.attribute(n).isNumeric()) {
                HashSet<Integer> uniqueValues = new HashSet();
                for (int i = 0; i < data.numInstances(); ++i) {
                    uniqueValues.add((int) (data.instance(i).value(att)));
                }
                List<Integer> dataValues = new ArrayList<>(uniqueValues);
                dataValues.sort((Integer o1, Integer o2) -> {
                    if (o1 > o2) return 1;
                    else return -1;                    
                });
                
                double[] infoGains = new double[dataValues.size() - 1];
                Instances[] tempInstances = new Instances[dataValues.size() - 1];
                for (int i = 0; i < dataValues.size() - 1; ++i) {
                    tempInstances[i] = setAttributeThreshold(data, att, dataValues.get(i));
                    infoGains[i] = calculateIG(tempInstances[i], tempInstances[i].attribute(att.name()));
                }
                data = new Instances(tempInstances[Util.indexOfMax(infoGains)]);
            }
        }
        return data;
    }
}
