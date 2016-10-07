/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package myclassifier;
import java.util.Scanner;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 * @author mochamadtry
 */
public class MyClassifier {

    /** 
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
        String nameOfFile, nameOfFileTest; 
        Classifier classifier; 
        Instances dataSet;
        Instances dataTest;
        boolean prune; 
        Scanner input = new Scanner(System.in);  
        System.out.println("My Classifier");
        System.out.print("Masukkan File Data Train: ");
        String option; 
        
        //Baca input file 
        Scanner scan = new Scanner(System.in); 
        nameOfFile= scan.nextLine(); 
        try {
            //Baca File arff
            dataSet = wekaCode.readFileArff(nameOfFile);
            
            //Remove Attributes
            System.out.print("Apakah akan removes attributes (Ya/Tidak) : ");
            option = scan.next(); 
            if (option.equals("Ya")){
                System.out.println("Masukkan index yang akan dihapus : ");
                String removeAtr = scan.next(); 
                Instances FilterData = wekaCode.removeAttributes(dataSet, removeAtr); 
                System.out.println(FilterData); 
            }
            
            //Resample
            System.out.print("Apakah akan melakukan sampling (Ya/Tidak) : ");
            option = scan.next();
            if (option.equals("Ya")){
                Instances FilterData = wekaCode.resampleData(dataSet); 
                System.out.println(FilterData);
            }
            
            
            //Build Classifier
            System.out.println("Tuliskan model classifier : 0.BAYES / 1.ID3 / 2.J48 / 3.MyID3 /4.MyJ48 ");
            int classifierType = scan.nextInt();
            classifier = wekaCode.buildClassifier(dataSet, classifierType, true);
            System.out.println(classifier.getClass());
            
            //Given test set 
            System.out.println("====GIVEN TEST SET====");
            System.out.print("Masukkan File Data Test: ");
            nameOfFileTest= scan.next();
            dataTest = wekaCode.readFileArff(nameOfFileTest);
            wekaCode.testingTestSet(dataSet, classifier, dataTest);
            
            
            //10-fold Validation 
            System.out.println("====10-FOLD VALIDATION====");
            wekaCode.foldValidation(dataSet, classifier);
            
            //percentage split
            System.out.println("====PERCENTAGE SPLIT====");
            System.out.print("Masukkan persen split: ");
            float pencentSplit = scan.nextFloat();
            wekaCode.percentageSplit(dataSet, classifier, pencentSplit);
            
            //Classify Unseen Data 
            System.out.println("====ALL DATA====");
            Instances LabelData = wekaCode.classifyUnseenData(classifier, dataSet);
            System.out.println(LabelData);
            System.out.println("====CLASSIFY UNSEEN DATA====");
            System.out.print("\nMasukkan nilai atribut pisahkan dengan spasi: ");
            String in = input.nextLine();
            String[] attributes = in.split(" ");
            wekaCode.classifyUnseenData(attributes, classifier, dataSet);
            
            //Load and Save Model\
            System.out.print("Apakah akan removes attributes (Ya/Tidak) : ");
            option = scan.next(); 
            if (option.equals("Ya")){
                String FileName = ".model" ;
                classifier = wekaCode.loadModel(FileName);
                wekaCode.saveModel(FileName, classifier);
            }
            
            
         } catch (Exception ex) {
            Logger.getLogger(MyClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
        
    
    
}
