/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import weka.classifiers.Classifier;
import weka.core.*;
import weka.core.Capabilities.Capability;

import java.util.*;

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
        capabilities.enable(Capability.BINARY_CLASS);
        capabilities.enable(Capability.NOMINAL_CLASS);

        capabilities.setMinimumNumberInstances(0);
        return capabilities;
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {
        // Mengecek dapatkah classifier menghandle data
        getCapabilities().testWithFail(instances);

        
        // cek apakah class nominal jika tidak exception 
        if (!instances.classAttribute().isNominal()) {
            throw new Exception("Class is not nominal");
        }
        
        for (int i = 0; i < instances.numAttributes(); i++) {
            Attribute attr = instances.attribute(i);
            // cek apakah attribut nominal jika tidak exception
            if (!attr.isNominal()) {
                throw new Exception("Attribute is not nominal");
            }
            
            for (int j = 0; j < instances.numInstances(); j++) {
                Instance inst = instances.instance(j);
                // cek apakah ada missing value jika tidak exception
                if (inst.isMissing(attr)) {
                    throw new Exception("Missing value detected");
                }
            }
        }
        
        Instances i = new Instances(instances);
        // menghapus instance dengan missing class
        i.deleteWithMissingClass();
        buildTree(i);
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
                infoGains[attr.index()] = Util.calculateIG(instances, attr);
            }
            
            splitAttribute = instances.attribute(Util.indexOfMax(infoGains));

            // Jika IG max = 0, buat daun dengan label kelas mayoritas
            // Jika tidak, buat successor
            if (Util.equalValue(infoGains[splitAttribute.index()], 0)) {
                splitAttribute = null;
                classDistribution = new double[instances.numClasses()];

                for (int i = 0; i < instances.numInstances(); i++) {
                    Instance inst = (Instance) instances.instance(i);
                    classDistribution[(int) inst.classValue()]++;
                }

                Util.normalizeClassDistribution(classDistribution);
                classValue = Util.indexOfMax(infoGains);
                classAttribute = instances.classAttribute();
            } else {
                Instances[] splitData = Util.splitData(instances, splitAttribute);
                successors = new MyID3[splitAttribute.numValues()];
                for (int j = 0; j < splitAttribute.numValues(); j++) {
                    successors[j] = new MyID3();
                    successors[j].buildTree(splitData[j]);
                }
            }
        }
    }
    
    @Override
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if (instance.hasMissingValue())
            throw new NoSupportForMissingValuesException("classifier.MyID3: This classifier can not handle missing value");
        if (splitAttribute == null)
            return classValue;
        else
            return successors[(int) instance.value(splitAttribute)].classifyInstance(instance);
    }
}
