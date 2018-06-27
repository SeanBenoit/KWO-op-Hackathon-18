import org.apache.spark.sql.SparkSession

val spark = SparkSession.builder().appName("Spark SQL basic example").config("spark.some.config.option", "some-value").getOrCreate()

import spark.implicits._

val df = spark.read.json("publicJSONFeed.json")
df.show()
val flattened = df.select($"vehicle", explode($"vehicle").as("vehicle_flat"))
flattened.show()