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
    resample.setInputFormat(randData)

    val filtered = Filter.useFilter(randData, resample).enumerateInstances()

    var cnt = 0
    var refimprove = 0
    while(filtered.hasMoreElements) {
      cnt+=1
      val e = filtered.nextElement().asInstanceOf[Instance]
      val v = e.value(randData.numAttributes - 1)
      if (v == 0.0)
        refimprove += 1


      println(v)
    }

    println(randData.attributeStats(randData.numAttributes - 1))

    val att = randData.attribute(randData.numAttributes - 1)
    val refImpIdx = att.indexOfValue("Refimprove")

    val outcomes = getRBFCandidates.par.map { cand =>
      val eTestb = new Evaluation(randData)
      val c1Model: Classifier = new LibSVM()


      c1Model.setOptions(weka.core.Utils.splitOptions(cand.params))

      // using seed = 1
      val rand = new Random(1)
      eTestb.crossValidateModel(c1Model, randData, folds, rand)

      println(cand)
      println(eTestb.toClassDetailsString)
      (eTestb, cand.cc)
    }

    // pick the best one
    print("best C: " + outcomes.maxBy(_._1.fMeasure(refImpIdx))._2)
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
