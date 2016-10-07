/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

import java.util.Arrays;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;
import weka.classifiers.Evaluation;
import java.util.Random;
import weka.core.Instance;
import weka.core.Attribute;


/**
 *
 * @author mochamadtry
 */
public class wekaCode {
    public static final int BAYES = 0;
    public static final int ID3 = 1;
    public static final int J48 = 2;
    public static final int MyID3 = 3;
    public static final int MyJ48 = 4;
    //public static Instances data;
    //public static Classifier classifier;
    
    //Baca File .arff, .csv etc
    public static Instances readFileArff(String fileName) throws Exception{
        //http://weka.sourceforge.net/doc.stable/weka/core/Instances.html
        //membaca semua instances dari file .arff, .csv
        DataSource source = new DataSource(fileName);
        Instances dataSet = source.getDataSet();
        //set atribut terakhir sebagai kelas 
        if (dataSet.classIndex()== -1)
             dataSet.setClassIndex(dataSet.numAttributes() - 1); //Make the last attribute be the class
        return dataSet; 
    }
    
    public static Instances removeAttributes(Instances data, String attribute) throws Exception{
        Remove remove = new Remove(); 
        remove.setAttributeIndices(attribute); //Set which attributes are to be deleted (or kept if invert is true)
        remove.setInputFormat(data); //Sets the format of the input instances.
        Instances filterData = Filter.useFilter(data, remove); //Filters an entire set of instances through a filter and returns the new set.
        return filterData;    
    }
    
    //Filter : resample 
    public static Instances resampleData(Instances data) throws Exception{
        Resample resample = new Resample(); 
        resample.setInputFormat(data);
        Instances filterData = Filter.useFilter(data, resample);
        return filterData; 
    }
    
    //Build Classifier
    public static Classifier buildClassifier(Instances dataSet, int classifierType, boolean prune) throws Exception{
        Classifier classifier = null;
        if (classifierType == BAYES){
            classifier = new NaiveBayes(); 
            classifier.buildClassifier(dataSet);
        }
        else if(classifierType == ID3){
            classifier = new Id3();
            classifier.buildClassifier(dataSet);
        }
        else if(classifierType == J48){
            classifier = new J48();
            classifier.buildClassifier(dataSet);
        } 
        else if(classifierType == MyID3){
            classifier = new MyID3();
            classifier.buildClassifier(dataSet);
        }
        else if(classifierType == MyJ48){
            MyJ48 j48 = new MyJ48();
            j48.setPruning(prune);
            classifier = j48;
            classifier.buildClassifier(dataSet);
            
        }
        return classifier;
    }
    
    //Testing model given test set 
    public static void testingTestSet(Instances dataSet, Classifier classifiers, Instances testSet) throws Exception{
        Evaluation evaluation = new Evaluation(dataSet); 
        evaluation.evaluateModel(classifiers, testSet); //Evaluates the classifier on a given set of instances.
        System.out.println(evaluation.toSummaryString("\n Testing Model given Test Set ", false));
        System.out.println(evaluation.toClassDetailsString());
        
        
        
    }
    
    //10-fold cross validation
     public static void foldValidation(Instances dataSet, Classifier classifiers) throws Exception{
        Evaluation evaluation = new Evaluation(dataSet); 
        evaluation.crossValidateModel(classifiers, dataSet, 10, new Random(1)); //Evaluates the classifier on a given set of instances.
        System.out.println(evaluation.toSummaryString("\n 10-fold cross validation", false));  
        System.out.println(evaluation.toMatrixString("\n Confusion Matrix"));
        
    }
     
    //Percentage Split
    public static void percentageSplit (Instances data, Classifier classifiers, float percentSplit) throws Exception{
        //Split data jadi training data dan test data 
        int jumlahDataTrain = Math.round(data.numInstances() * (percentSplit/100));
        int jumlahDataTest = data.numInstances() - jumlahDataTrain; 
        Instances dataTrain = new Instances(data, 0, jumlahDataTrain);
        Instances dataTest = new Instances(data, jumlahDataTrain, jumlahDataTest);
        
        //Evaluate 
        classifiers.buildClassifier(dataTrain);
        testingTestSet(dataTrain, classifiers, dataTest);    
    }
    
    //Save and Load Model 
    public static Classifier loadModel(String fileName) throws Exception {
        return (Classifier) SerializationHelper.read(fileName);
    }
    
    public static void saveModel(String fileName, Classifier classifiers) throws Exception {
        SerializationHelper.write(fileName, classifiers);
    }

    
    //Lihat semua data 
    public static Instances classifyUnseenData (Classifier classifiers, Instances dataSet) throws Exception{
        Instances labeledData = new Instances(dataSet);
        // labeling data
        for (int i = 0; i < labeledData.numInstances(); i++) {
            double clsLabel = classifiers.classifyInstance(dataSet.instance(i));
            labeledData.instance(i).setClassValue(clsLabel);
        }
        return labeledData;
    }
    
    //Using model to classify one unseen data(input data)
    public static void classifyUnseenData(String[] attributes, Classifier classifiers, Instances data) throws Exception {
        Instance newInstance = new Instance(data.numAttributes());
        newInstance.setDataset(data);
        for (int i = 0; i < data.numAttributes()-1; i++) {
            if(Attribute.NUMERIC == data.attribute(i).type()){
                Double value = Double.valueOf(attributes[i]);
                newInstance.setValue(i, value);
            } else {
                newInstance.setValue(i, attributes[i]);
            }
        }
        
        double clsLabel = classifiers.classifyInstance(newInstance);
        newInstance.setClassValue(clsLabel);
        
        String result = data.classAttribute().value((int) clsLabel);
        
        System.out.println("Hasil Classify Unseen Data Adalah: " + result);
    }
    
    
}
