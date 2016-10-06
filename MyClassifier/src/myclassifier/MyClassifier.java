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
    public static final int MODE_TRAIN = 0;
    public static final int MODE_CLASSIFY = 1;
    public static final int MODE_CROSS_VALIDATE = 2;
    public static final int MODE_EVALUATE_SPLIT = 3;
    public static final int MODE_VALIDATE_TEST_SET = 4;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        
        String nameOfFile; 
        Classifier classifier; 
        Instances dataSet; 
        boolean prune; 
        
        //Baca input file 
        Scanner scan = new Scanner(System.in); 
        nameOfFile= scan.nextLine(); 
        try {
            //Baca File arff
            dataSet = wekaCode.readFileArff(nameOfFile);
            System.out.println(dataSet.firstInstance());
            
            //Remove Attributes
            System.out.println("Masukkan index yang akan dihapus : ");
            //String removeAtr = scan.next(); 
            //Instances FilterData = wekaCode.removeAttributes(dataSet, removeAtr); 
            //System.out.println(FilterData);
            
            //Resample 
            //FilterData = wekaCode.resampleData(dataSet); 
            //System.out.println(FilterData);
            
            //Build Classifier
            System.out.println("Tuliskan model classifier : 0.BAYES / 1.ID3 / 2.J48 / 3.MyID3 /4.MyJ48 ");
            int classifierType = scan.nextInt();
            classifier = wekaCode.buildClassifier(dataSet, classifierType, true);
            System.out.println(classifier.getClass());
            
            //Given test set 
            
            //10-fold Validation 
            wekaCode.foldValidation(dataSet, classifier);
            
            //percentage split
            System.out.println(dataSet.numInstances());
            System.out.println("Masukkan persen split: ");
            float pencentSplit = scan.nextFloat();
            wekaCode.percentageSplit(dataSet, classifier, pencentSplit);
            
            
            
            
            
            
            /*String dataFile = null;
            String testFile = null;
            String modelName;
            boolean prune = false;
            Classifier classifier = null;
            int algorithm = -1;
            int mode = -1;
            
            for(int i=0; i<args.length; i+=2){
            if(args[i].equals("-data")){
            dataFile = args[i+1];
            }else if(args[i].equals("-al")){
            if(args[i+1].equals("bayes")){
            algorithm = wekaCode.BAYES;
            }else if(args[i+1].equals("id3")){
            algorithm = wekaCode.ID3;
            }else if(args[i+1].equals("j48")){
            algorithm = wekaCode.J48;
            }
            }else if(args[i].equals("-mode")){
            if(args[i+1].equals("train")){
            mode = MyClassifier.MODE_TRAIN;
            }else if(args[i+1].equals("classify")){
            mode = MyClassifier.MODE_CLASSIFY;
            }else if(args[i+1].equals("crossvalidate")){
            mode = MyClassifier.MODE_CROSS_VALIDATE;
            }else if(args[i+1].equals("evaluatesplit")){
            mode = MyClassifier.MODE_EVALUATE_SPLIT;
            }
            }else if(args[i].equals("-test")){
            testFile = args[i+1];
            }else if(args[i].equals("-model")){
            try {
            classifier = wekaCode.loadModel(args[i + 1]);
            } catch (Exception e) {
            e.printStackTrace();
            }
            }else if(args[i].equals("-prune")){
            prune = true;
            }
            }
            try {
            if (mode == MyClassifier.MODE_TRAIN) {
            Instances data = wekaCode.readFileArff(dataFile);
            classifier = wekaCode.buildClassifier(data, algorithm, prune);
            wekaCode.saveModel(dataFile.replace(".arff", ".model"), classifier);
            } else if (mode == MyClassifier.MODE_CLASSIFY) {
            Instances dataTest = wekaCode.readFileArff(testFile);
            Instances classifiedInstances = wekaCode.classifyUnseenData(classifier, dataTest);
            } else if (mode == MyClassifier.MODE_CROSS_VALIDATE) {
            Instances data = wekaCode.readFileArff(dataFile);
            wekaCode.foldValidation(data, classifier);
            }else if(mode == MyClassifier.MODE_EVALUATE_SPLIT){
            Instances data = wekaCode.readFileArff(dataFile);
            wekaCode.percentageSplit(data, classifier, 30);
            }
            
            //            debugMode();
            }catch(Exception e){
            e.printStackTrace();
            }
            }*/
            
            /*public static void debugMode() throws Exception {
            Instances data = wekaCode.readFileArff("data/cpu.arff");
            Classifier classifier = wekaCode.buildClassifier(data, WekaIFace.MY_J48, false);
            WekaIFace.crossValidate(data, classifier);
        }*/ } catch (Exception ex) {
            Logger.getLogger(MyClassifier.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
        
    
    
}
