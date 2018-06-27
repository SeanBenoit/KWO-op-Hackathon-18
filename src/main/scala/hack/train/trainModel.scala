package hack.train

import org.apache.log4j._
import org.apache.hadoop.fs._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.rogach.scallop._
import scala.math._
import org.apache.spark.sql.SparkSession

class ConfA(args: Seq[String]) extends ScallopConf(args) {
  mainOptions = Seq(input, output)
  val input = opt[String](descr = "input path", required = true)
  val output = opt[String](descr = "output path", required = true)
  verify()
}

object trainModel {
  val log = Logger.getLogger(getClass().getName())

  def main(argv: Array[String]) {
    val spark = SparkSession.builder().appName("name").getOrCreate()
    import spark.implicits._

    val args = new ConfA(argv)

    log.info("Input: " + args.input())
    log.info("Output: " + args.output())

    val sc = spark.sparkContext

    val df = spark.read.json(args.input())

    val outputDir = new Path(args.output())
    FileSystem.get(sc.hadoopConfiguration).delete(outputDir, true)

    df.write.json(args.output())
  }
}
