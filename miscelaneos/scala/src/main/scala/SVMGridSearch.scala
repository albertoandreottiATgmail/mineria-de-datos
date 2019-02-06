package unsl.alberto
import weka.classifiers.Classifier
import weka.classifiers.Evaluation
import weka.classifiers.functions.LibSVM
import weka.core.{Instance, Instances}
import java.io.BufferedReader
import java.io.FileReader
import java.util.Random

import weka.filters.Filter
import weka.filters.supervised.instance.Resample


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

    val resample = new Resample()

    resample.setBiasToUniformClass(1.0)
    resample.setSampleSizePercent(2.0 / 13.0  * 100)
    resample.setInputFormat(randData)

    val subsampled = Filter.useFilter(randData, resample)
    subsampled.stratify(folds)

    val dataset = subsampled

    val att = randData.attribute(randData.numAttributes - 1)
    val refImpIdx = att.indexOfValue("Refimprove")

    //val outcomes = getRBFCandidates(3, 15, 2).par.map { cand =>
    val outcomes = getLinearCandidates(3, 15, 2).par.map { cand =>
      val eTestb = new Evaluation(dataset)
      val c1Model: Classifier = new LibSVM()

      c1Model.setOptions(weka.core.Utils.splitOptions(cand.params))

      // using seed = 1
      val rand = new Random(1)
      eTestb.crossValidateModel(c1Model, dataset, folds, rand)

      println(eTestb.toClassDetailsString)
      (eTestb, cand.cc)
    }

    outcomes.foreach{ case (etest, cc) =>
      println(cc, etest.fMeasure(refImpIdx))
    }
    // pick the best one
    println("best C: " + outcomes.maxBy(_._1.fMeasure(refImpIdx))._2)
    println("f score: " + outcomes.maxBy(_._1.fMeasure(refImpIdx))._1.fMeasure(refImpIdx))
  }

  def getRBFCandidates(start:Int, end:Int, step:Int) = {
    for (cg <- Range(-1, 3 + 1, 2);cc <- Range(start, end + 1, step))
        yield RBFParamSet(cg, cc)
  }

  def getLinearCandidates(start:Int, end:Int, step:Int) = {
    for (cc <- Range(start, end + 1, step))
      yield LinearParamSet(cc)
  }

}

case class RBFParamSet(cg:Int, cc:Int) {
  override def toString: String = {s"cg: 2^$cg, cc:2^$cc"}
  val params = s"-S 0 -K 2 -D 3 -G ${Math.pow(2, cg)} " +
    s"-R 0.0 -N 0.5 -M 40.0 -C ${Math.pow(2, cc)} -E 0.0010 -P 0.1 -Z"
}

case class LinearParamSet(cc:Int) {
  override def toString: String = {s"cc:2^$cc"}
  val params = s"-S 0 -K 0 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C ${Math.pow(2, cc)} -E 0.0010 -P 0.1 -Z"
}
