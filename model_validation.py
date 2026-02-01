from pyspark.sql import SparkSession, functions as F


BUCKET = "epwa-delays-project"

SILVER_PATH = f"s3a://{BUCKET}/silver/daily/"

PRED_PATH   = f"s3a://{BUCKET}/gold/model/v1/predictions_full"

OUT_DAILY   = f"s3a://{BUCKET}/gold/model/v1/validation_daily"
OUT_MONTHLY = f"s3a://{BUCKET}/gold/model/v1/validation_monthly"

AIRPORT_CODE = "EPWA"

=
spark = (
    SparkSession.builder
    .appName("MODEL-VALIDATION-V1")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print(f"\n[INFO] SILVER_PATH = {SILVER_PATH}")
print(f"[INFO] PRED_PATH   = {PRED_PATH}")
print(f"[INFO] OUT_DAILY   = {OUT_DAILY}")
print(f"[INFO] OUT_MONTHLY = {OUT_MONTHLY}\n")

silver_raw = spark.read.parquet(SILVER_PATH)
pred_raw   = spark.read.parquet(PRED_PATH)

print("[INFO] Silver columns:", silver_raw.columns)
print("[INFO] Pred columns:", pred_raw.columns)


silver = silver_raw

if "AIRPORT" in silver.columns:
    silver = silver.filter(F.col("AIRPORT") == F.lit(AIRPORT_CODE))
else:
    raise RuntimeError("Brak kolumny AIRPORT w silver – sprawdź etl.py")

required_silver = ["DATE", "YEAR", "MONTH", "AVG_ARR_DELAY_MIN", "AIRPORT"]
missing_s = [c for c in required_silver if c not in silver.columns]
if missing_s:
    raise RuntimeError(f"Brak wymaganych kolumn w silver: {missing_s}")

silver = silver.select(*required_silver)


silver = silver.withColumn("DATE", F.to_date(F.col("DATE")))


silver = silver.dropDuplicates(["AIRPORT", "DATE", "YEAR", "MONTH"])


pred = pred_raw

required_pred = ["DATE", "prediction"]
missing_p = [c for c in required_pred if c not in pred.columns]
if missing_p:
    raise RuntimeError(
        f"Brak wymaganych kolumn w predictions_full: {missing_p}. "
    )

pred = pred.select("DATE", "prediction")
pred = pred.withColumn("DATE", F.to_date(F.col("DATE")))
pred = pred.withColumn("prediction", F.col("prediction").cast("double"))


pred = pred.dropDuplicates(["DATE"])


s = silver.alias("s")
p = pred.alias("p")

df = (
    s.join(p, on="DATE", how="inner")
     .select(
         F.col("s.DATE").alias("DATE"),
         F.col("s.YEAR").cast("int").alias("YEAR"),
         F.col("s.MONTH").cast("int").alias("MONTH"),
         F.col("s.AVG_ARR_DELAY_MIN").cast("double").alias("AVG_ARR_DELAY_MIN"),
         F.col("p.prediction").cast("double").alias("prediction")
     )
)


print("\n[CHECK] silver rows:", silver.count())
print("[CHECK] pred rows  :", pred.count())
print("[CHECK] joined rows:", df.count())

if df.count() == 0:
    raise RuntimeError("JOIN dał 0 wierszy")


df = (
    df
    .withColumn("error_min", F.col("prediction") - F.col("AVG_ARR_DELAY_MIN"))
    .withColumn("abs_error_min", F.abs(F.col("error_min")))
)

print("\nSAMPLE (daily validation)")
df.orderBy("DATE").show(10, False)



monthly = (
    df.groupBy("YEAR", "MONTH")
      .agg(
          F.count("*").alias("days_count"),
          F.avg("AVG_ARR_DELAY_MIN").alias("real_avg_delay_mean"),
          F.avg("prediction").alias("pred_avg_delay_mean"),
          F.avg("abs_error_min").alias("mae_mean"),
          F.sqrt(F.avg(F.pow(F.col("error_min"), F.lit(2.0)))).alias("rmse")
      )
      .orderBy("YEAR", "MONTH")
)

print("\n WALIDACJA MIESIĘCZNA")
monthly.show(200, False)


(
    df.select(
        "DATE", "YEAR", "MONTH",
        "AVG_ARR_DELAY_MIN", "prediction",
        "error_min", "abs_error_min"
    )
    .write.mode("overwrite")
    .parquet(OUT_DAILY)
)

(
    monthly.write.mode("overwrite")
    .parquet(OUT_MONTHLY)
)

print("\n[OK] Walidacja zapisana do Gold ")
print(f"[OK] Daily:   {OUT_DAILY}")
print(f"[OK] Monthly: {OUT_MONTHLY}\n")

spark.stop()
