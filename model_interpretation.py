from pyspark.sql import SparkSession, functions as F

BUCKET = "epwa-delays-project"
IMP_PATH = f"s3a://{BUCKET}/gold/model/v1/feature_importance"
OUT_PATH = f"s3a://{BUCKET}/gold/model/v1/interpretation"

spark = SparkSession.builder.appName("MODEL-INTERPRETATION").getOrCreate()

df = spark.read.parquet(IMP_PATH)


total = df.agg(F.sum("importance").alias("total")).collect()[0]["total"]

df = (
    df.withColumn("importance_pct", F.col("importance") / F.lit(total) * 100)
      .orderBy(F.desc("importance_pct"))
)

print("\n FEATURE IMPORTANCE (%)")
df.show(30, False)

df.write.mode("overwrite").parquet(OUT_PATH)

spark.stop()
print("\nZapisano interpretację do Gold")
