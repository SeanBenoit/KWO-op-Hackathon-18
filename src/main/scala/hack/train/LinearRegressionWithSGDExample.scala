/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package hack.train

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
// $example on$
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.feature.Normalizer
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.classification.{NaiveBayes, NaiveBayesModel}

// $example off$

@deprecated("Use ml.regression.LinearRegression or LBFGS", "2.0.0")
object LinearRegressionWithSGDExample {
  val log = Logger.getLogger(getClass().getName())

  def convertToSeconds(t: String): Integer = {
    val split = t.split(":")
    val seconds = split(0).toInt*3600 + split(1).toInt*60 + split(2).toInt
    //val seconds = (split(0)+split(1)+split(2)).toInt
    seconds
  }

  def main(args: Array[String]): Unit = {
    // val conf = new SparkConf().setAppName("LinearRegressionWithSGDExample")
    // val sc = new SparkContext(conf)
    val spark: SparkSession = SparkSession.builder.getOrCreate
    val sc = spark.sparkContext

    val trips = sc.textFile("/data/ttc/trips.txt")
      .map(line => {
        val split = line.split(",")
        (split(2), split(0))
      })

    val tripMap = sc.broadcast(trips.collectAsMap)

    val stopTimes = sc.textFile("/data/ttc/stop_times.txt")
      .map(line => { 
        val split = line.split(",")
        (split(0), convertToSeconds(split(1)), split(3), tripMap.value.getOrElse(split(0), "routeIDnotFOund"))
      })
      .map{ case(tripId, arrival, stop, route) => LabeledPoint(arrival.toDouble, Vectors.dense(Array(tripId, stop, route).map(_.toDouble)))}
      .filter(x => x.label < 86400)

    val scaler2 = new StandardScaler(withMean = true, withStd = true).fit(stopTimes.map(x => x.features))

    val normStopTimes = stopTimes.map(x => (x.label, scaler2.transform(Vectors.dense(x.features.toArray))))
       .filter(x => x._1 < 86400)
       .map(x => LabeledPoint(x._1, x._2))
    //   .saveAsTextFile("/data/tmp/labeled")
    // // $example on$
    // // Load and parse the data
    // val data = sc.textFile("/data/mllib/ridge-data/lpsa.data")
    // val parsedData = data.map { line =>
    //   val parts = line.split(',')
    //   LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(' ').map(_.toDouble)))
    // }.cache()

    // Building the model
    val numIterations = 100
    val stepSize = 8
    val model = LinearRegressionWithSGD.train(normStopTimes, numIterations, stepSize)
    // val model = new LogisticRegressionWithLBFGS()
    //   .setNumClasses(86400)
    //   .run(normStopTimes)
    // val model = NaiveBayes.train(stopTimes, lambda = 1.0, modelType = "multinomial")

    // val predictionAndLabel = stopTimes.map(p => (model.predict(p.features), p.label))
    // val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / stopTimes.count()
    // Evaluate model on training examples and compute training error
    val valuesAndPreds = normStopTimes.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    // val metrics = new MulticlassMetrics(valuesAndPreds)
    // val accuracy = metrics.accuracy
    // log.info(s"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Accuracy = $accuracy")
    val MSE = valuesAndPreds.map{ case(v, p) => math.pow((v - p), 2) }.mean()
    log.info(s"############################################################################ training Mean Squared Error $MSE")

    val outputDir = new Path("/data/model/tmp")
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    // Save and load model
    model.save(sc, "/data/model/tmp/testModel")

    valuesAndPreds.saveAsTextFile("/data/model/tmp/predictions")
    // val sameModel = LinearRegressionModel.load(sc, "/data/model/tmp/scalaLinearRegressionWithSGDModel")
    // $example off$

    sc.stop()
  }
}
// scalastyle:on println