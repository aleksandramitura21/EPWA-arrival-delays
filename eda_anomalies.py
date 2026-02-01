from pyspark.sql import SparkSession, functions as F

BUCKET = "epwa-delays-project"
SILVER_PATH = f"s3a://{BUCKET}/silver/daily/"
GOLD_PATH = f"s3a://{BUCKET}/gold/eda/anomalies"

spark = (
    SparkSession.builder
    .appName("EPWA-EDA-ANOMALIES")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)

df = spark.read.parquet(SILVER_PATH)


if "AIRPORT" in df.columns:
    df = df.filter(F.col("AIRPORT") == "EPWA")


required = ["DATE", "TOTAL_DELAY_MIN", "AVG_ARR_DELAY_MIN"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Brakuje kolumn: {missing}")


p95 = df.approxQuantile("AVG_ARR_DELAY_MIN", [0.95], 0.01)[0]
p99 = df.approxQuantile("AVG_ARR_DELAY_MIN", [0.99], 0.01)[0]

print("95 percentyl avg delay:", p95)
print("99 percentyl avg delay:", p99)


top_avg = (
    df.orderBy(F.desc("AVG_ARR_DELAY_MIN"))
      .limit(30)
      .withColumn("anomaly_type", F.lit("TOP_AVG_DELAY"))
)


top_sum = (
    df.orderBy(F.desc("TOTAL_DELAY_MIN"))
      .limit(30)
      .withColumn("anomaly_type", F.lit("TOP_TOTAL_DELAY"))
)


outliers = (
    df.filter(F.col("AVG_ARR_DELAY_MIN") >= F.lit(p95))
      .withColumn("anomaly_type", F.lit("P95_OUTLIER"))
)


anomalies = top_avg.unionByName(top_sum).unionByName(outliers)


anomalies = anomalies.withColumn(
    "severity",
    F.when(F.col("AVG_ARR_DELAY_MIN") >= p99, "EXTREME")
     .when(F.col("AVG_ARR_DELAY_MIN") >= p95, "HIGH")
     .otherwise("MODERATE")
)

anomalies = anomalies.orderBy(F.desc("AVG_ARR_DELAY_MIN"))

print("\n PRZYKŁADY")
anomalies.select(
    "DATE",
    "AVG_ARR_DELAY_MIN",
    "TOTAL_DELAY_MIN",
    "anomaly_type",
    "severity"
).show(30, False)


anomalies.write.mode("overwrite").parquet(GOLD_PATH)

print("\nZapisano do:")
print(GOLD_PATH)

spark.stop()
)
