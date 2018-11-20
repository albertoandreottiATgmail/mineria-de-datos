import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;

public class curso_MD_cross_val_SVM_grid_search {
	
	public static void main(String[] args) throws Exception{
		
		/**
		 * This program receives only one input parameter, namely:
		 * 
		 * 0.  The path to the ARFF file where it is the set to be used for cross-validation
		 * 
		 * */
		 
		 int folds = 10;        // the number of folds to generate, >=2
		 		 
		 // Set containing the samples
		 BufferedReader reader = new BufferedReader(new FileReader(args[0]));
		 Instances randData = new Instances(reader);
		 reader.close();
		 randData.setClassIndex(randData.numAttributes() - 1);
					
		 randData.stratify(folds); 						// The class attribute is nominal
		 
		 //for(int d=2;d <= 5; d++){//Este for se comenta para el kernel lineal y para RBF
		 //int d = 3;
		 	//for(int r = 0;r <= 1;r++){//Este for se comenta para el kernel lineal y para RBF
		 	//int r = 1;
		 		for(/*int cg = -15*/int cg = -3; cg <= 3; cg+=2){//Este for se comenta para el kernel lineal
		 		//int cg = 1; 
		      		for(/*int cc = -5*/ int cc = -1; cc <= 15; cc+=2){
		 			//int cc = 1;
		      	    		  				
				      		
		      				//K = 0 -> linear 
		      				//String parameters = new String("-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C ").concat(Double.toString(Math.pow(2, cc))).concat(" -E 0.0010 -P 0.1 -Z");
		      				
		      				//K = 1 -> polinomial
		      				//String parameters = new String("-S 0 -K 1 -D ".concat(Integer.toString(d)).concat(" -G ").concat(Double.toString(Math.pow(2, cg))).concat(" -R ").concat(Integer.toString(r)).concat(" -N 0.5 -M 40.0 -C ").concat(Double.toString(Math.pow(2, cc))).concat(" -E 0.0010 -P 0.1 -Z"));
		      				
		      				//K = 2 -> RBF 
	      					String parameters = new String("-S 0 -K 2 -D 3 -G ".concat(Double.toString(Math.pow(2, cg))).concat(" -R 0.0 -N 0.5 -M 40.0 -C ").concat(Double.toString(Math.pow(2, cc))).concat(" -E 0.0010 -P 0.1 -Z"));
		      			
		       				Evaluation eTestb = new Evaluation(randData);		
		       				
		       				Classifier c1Model = (Classifier) new LibSVM();
		       				c1Model.setOptions(weka.core.Utils.splitOptions(parameters));		 
		       					
		       				Random rand = new Random(1);  // using seed = 1
		       				 
		       				System.out.println("");
		       				System.out.println("Parameters = " + parameters);
		       				System.out.println("");
		       				//System.out.println("*** C = 2^"+ cc +" Gamma = 2^"+ cg + " D = " + d + " R = " + r + " ***");
	        				System.out.println("*** C = 2^"+ cc +" Gamma = 2^"+ cg +" ***");
		       				//System.out.println("*** C = 2^"+ cc +" ***");
	        				System.out.println("");
		       				 
		       				 eTestb.crossValidateModel(c1Model, randData, folds, rand);
		       				 System.out.println(eTestb.toClassDetailsString());
		       				
		       							       				
		      		}// end for(int cc = -5; cc <=15; cc+=2)
		 		}// end for(int cg = -15; cg <=3; cg+=2)
		 	//}//end for(int r = 0;r <= 1;r++) 
		//}// end for(int d=2;d <= 5; d++)
	}
}
