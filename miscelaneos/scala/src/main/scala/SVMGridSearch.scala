package unsl.alberto
import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.functions.LibSVM
import weka.core.Instances
import java.io.BufferedReader
import java.io.FileReader
import java.util.Random


object SVMGridSearch {

  /**
    * This program receives only one input parameter, namely:
    *
    * 0.  The path to the ARFF file where it is the set to be used for cross-validation
    *
    * */

  def main(args: Array[String]) {
    // the number of folds to generate, >=2
    val folds = 10

    // Set containing the samples
    val reader = new BufferedReader(new FileReader(args(0)))
    val randData = new Instances(reader)
    reader.close()

    randData.setClassIndex(randData.numAttributes - 1)
    randData.stratify(folds)

    val outcomes = getRBFCandidates.par.map { cand =>
      val eTestb = new Evaluation(randData)
      val c1Model: Classifier = new LibSVM()
      c1Model.setOptions(weka.core.Utils.splitOptions(cand.params))

      // using seed = 1
      val rand = new Random(1)
      eTestb.crossValidateModel(c1Model, randData, folds, rand)

      println(cand)
      println(eTestb.toClassDetailsString)
    }
  }

  def getRBFCandidates = {
    for (cg <- Range(-3, 3, 2);cc <- Range(5, 11, 2))
        yield RBFParamSet(cg, cc)
  }

}

case class RBFParamSet(cg:Int, cc:Int) {
  override def toString: String = {s"cg: 2^$cg, cc:2^$cc"}
  val params = s"-S 0 -K 2 -D 3 -G ${Math.pow(2, cg)} " +
    s"-R 0.0 -N 0.5 -M 40.0 -C ${Math.pow(2, cc)} -E 0.0010 -P 0.1 -Z"
}
