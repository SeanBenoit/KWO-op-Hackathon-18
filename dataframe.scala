import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types._

val spark = SparkSession.builder().appName("Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()

import spark.implicits._

val df = spark.read.json("publicJSONFeed.json")
def children(colname: String, df: DataFrame) = {
  val parent = df.schema.fields.filter(_.name == colname).head
  val fields = parent.dataType match {
    case x: StructType => x.fields
    case _ => Array.empty[StructField]
  }
  fields.map(x => col(s"$colname.${x.name}"))
}

val newDF = df.select(explode($"vehicle").as("vehicle"))
newDF.select(children("vehicle", newDF): _*).printSchema
newDF.show()
