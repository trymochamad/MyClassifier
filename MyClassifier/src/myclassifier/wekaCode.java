/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;

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


/**
 *
 * @author mochamadtry
 */
public class wekaCode {
    
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
    
    
    
}
