import weka.classifiers.Classifier;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class curso_MD_cross_val_alt_SVM {
	
	public static void main(String[] args) throws Exception{
		
		/**
		 * This program receives 1 input parameters, namely:
		 * 
		 * 0.  The path to the ARFF file where it is the train set for classifier.
		 * 
		 * */
		
		 int seed = 0;          // the seed for randomizing the data
		 int folds = 10;        // the number of folds to generate, >=2
		 		
		 BufferedReader reader2 = new BufferedReader(new FileReader(args[0]));
		 Instances randData = new Instances(reader2);
		 reader2.close();
		 randData.setClassIndex(randData.numAttributes() - 1);
		 
		 Random rand = new Random(seed);   				// create seeded number generator
		 randData.randomize(rand);         				// randomize data with number generator
			
		 randData.stratify(folds); 						// The class attribute is nominal		
    		
		 int cg = 1, cc = 1, r = 1, d = 2;
		 int tp=0,fn=0,fp=0,tn=0;
		 float precision_acum=0.0F,recall_acum=0.0F,f11_acum=0.0F;
		 
		 for (int n = 0; n < folds; n++) {
    			Instances train = randData.trainCV(folds, n);
    			Instances test = randData.testCV(folds, n);
	   
    			System.out.println("=== Cross-validation N° " + (n+1) + " ===");
    			System.out.println("=== train = " + train.numInstances() + " ===");
    			System.out.println("=== test = " + test.numInstances() + " ===");
    			
    			
	      		
  				//K = 0 -> linear 
  				//String parameters = new String("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C ").concat(Double.toString(Math.pow(2, cc))).concat(" -E 0.0010 -P 0.1 -Z");
  				
  				//K = 1 -> polinomial
  				String parameters = new String("-S 0 -K 1 -D ".concat(Integer.toString(d)).concat(" -G ").concat(Double.toString(Math.pow(2, cg))).concat(" -R ").concat(Integer.toString(r)).concat(" -N 0.5 -M 40.0 -C ").concat(Double.toString(Math.pow(2, cc))).concat(" -E 0.0010 -P 0.1 -Z"));
  				
  				//K = 2 -> RBF 
  				//String parameters = new String("-S 0 -K 2 -D 3 -G ".concat(Double.toString(Math.pow(2, cg))).concat(" -R 0.0 -N 0.5 -M 40.0 -C ").concat(Double.toString(Math.pow(2, cc))).concat(" -E 0.0010 -P 0.1 -Z"));
  				
   				Classifier c1Model = (Classifier) new LibSVM();
   				c1Model.setOptions(weka.core.Utils.splitOptions(parameters));
		
    			// train and make predictions
    			c1Model.buildClassifier(train);
       		
    			
    			
				for (int i = 0; i < test.numInstances(); i++) {
					   double pred = c1Model.classifyInstance(test.instance(i));
					   					   
					   if(test.classAttribute().value((int) test.instance(i).classValue()).equals("yes") && test.classAttribute().value((int) pred).equals("yes")) tp++;
						else if(test.classAttribute().value((int) test.instance(i).classValue()).equals("yes") && test.classAttribute().value((int) pred).equals("no")) fn++;
						else if(test.classAttribute().value((int) test.instance(i).classValue()).equals("no") && test.classAttribute().value((int) pred).equals("no")) tn++;
						else if(test.classAttribute().value((int) test.instance(i).classValue()).equals("no") && test.classAttribute().value((int) pred).equals("yes")) fp++;
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
				 
				precision_acum+=precision;
				recall_acum+=recall;
				f11_acum+=f11;
				 
				tp=0;
				fn=0;
				fp=0;
				tn=0;
				 
    	}//end for cross-validation
    		
    	System.out.println("### Avg. Precision = "+ (precision_acum/folds) +" ###");
		System.out.println("### Avg. Recall = "+ (recall_acum/folds) +" ###");
		System.out.println("### Avg. F1 = "+ (f11_acum/folds) +" ###");
    		
	}
}
