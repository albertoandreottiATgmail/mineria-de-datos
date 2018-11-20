import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.Remove;
import java.io.BufferedReader;
import java.io.FileReader;
import java.text.DecimalFormat;

public class curso_MD_test_set_SVM {
	
	public static void main(String[] args) throws Exception{
		
		/**
		 * This program receives 2 input parameters, namely:
		 * 
		 * 0.  The path to the ARFF file containing the training set.
		 * 1.  The path to the ARFF file containing the test set for classifier.
		 * 
		 * */
		
		 		 
		 // Training set for classifier
		 BufferedReader reader3 = new BufferedReader(new FileReader(args[0]));
		 Instances isTrainingSet2 = new Instances(reader3);
		 reader3.close();
		 isTrainingSet2.setClassIndex(isTrainingSet2.numAttributes() - 1);
		 
		 // Test set
		 BufferedReader reader2 = new BufferedReader(new FileReader(args[1]));
		 Instances isTestSet2 = new Instances(reader2);
		 reader2.close();
		 isTestSet2.setClassIndex(isTestSet2.numAttributes() - 1);
 			 
		 // The classifier is built as a linear SVM 	 
		 Classifier c2Model = (Classifier)new LibSVM();	
		
		 int cc = 15;			 
			 
		 String parameters = new String("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C ".concat(Double.toString(Math.pow(2, cc))).concat(" -E 0.0010 -P 0.1 -Z -B -seed 1"));

		 c2Model.setOptions(weka.core.Utils.splitOptions(parameters));

		 // train and make predictions
		 c2Model.buildClassifier(isTrainingSet2);
				 
		 System.out.println();
		 System.out.println("=== Classifier 2: evaluation on test set ===");
		 System.out.println("*** C = $2^{"+ cc +"}$ ***");
				 
		 int tp=0,fn=0,fp=0,tn=0;
		 for (int i = 0; i < isTestSet2.numInstances(); i++) {
			 double pred = c2Model.classifyInstance(isTestSet2.instance(i));
					   
			 if(isTestSet2.classAttribute().value((int) isTestSet2.instance(i).classValue()).equals("yes") && isTestSet2.classAttribute().value((int) pred).equals("yes")) tp++;
			 else if(isTestSet2.classAttribute().value((int) isTestSet2.instance(i).classValue()).equals("yes") && isTestSet2.classAttribute().value((int) pred).equals("no")) fn++;
			 else if(isTestSet2.classAttribute().value((int) isTestSet2.instance(i).classValue()).equals("no") && isTestSet2.classAttribute().value((int) pred).equals("no")) tn++;
			 else if(isTestSet2.classAttribute().value((int) isTestSet2.instance(i).classValue()).equals("no") && isTestSet2.classAttribute().value((int) pred).equals("yes")) fp++;
		 }
		
		 System.out.println("### Performance measures ###");
		 System.out.println("### TP = "+ tp +" ###");
		 System.out.println("### TN = "+ tn +" ###");
		 System.out.println("### FP = "+ fp +" ###");
		 System.out.println("### FN = "+ fn +" ###");
				 
		 int deno_temp=tp+fp;
		 float precision,recall,f11;   
		 if(deno_temp==0) precision=0;
		    else precision=(float)tp/(tp+fp);

		 deno_temp=tp+fn;
		 if(deno_temp==0) recall=0;
		 else recall=(float)tp/(tp+fn);

		        
		 if(precision!=0 && recall!=0)
		    f11=2*(precision*recall)/(precision+recall); 				
		 else f11=0;				 
				 
				 
		System.out.println("### Precision = "+ precision +" ###");
		System.out.println("### Recall = "+ recall +" ###");
		System.out.println("### F1 = "+ f11 +" ###");
				 
		//Evaluation eTestb = new Evaluation(isTrainingSet2);
		//eTestb.evaluateModel(c2Model, isTestSet2);
		//System.out.println(eTestb.toSummaryString());
		//System.out.println(eTestb.toClassDetailsString());
		//System.out.println(eTestb.toMatrixString());
	}
}