from pyspark.sql import SparkSession, functions as F

BUCKET = "epwa-delays-project"
SILVER_PATH = f"s3a://{BUCKET}/silver/daily/"
GOLD_PREFIX = f"s3a://{BUCKET}/gold/eda"


spark = (
    SparkSession.builder
    .appName("EPWA-EDA-BASIC")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .getOrCreate()
)


df = spark.read.parquet(SILVER_PATH)
from pyspark.sql import functions as F

print("\n TEST: LICZBA DNI W ROKU")
df.groupBy("YEAR") \
  .agg(
      F.count("*").alias("rows"),
      F.countDistinct("DATE").alias("unique_days")
  ) \
  .orderBy("YEAR") \
  .show(30, False)

print("Kolumny Silver:", df.columns)
print("Liczba rekordów:", df.count())


if "AIRPORT" in df.columns:
    df = df.filter(F.col("AIRPORT") == "EPWA")


required_cols = ["DATE", "YEAR", "MONTH", "DOW", "ARR_TOTAL_FLIGHTS", "TOTAL_DELAY_MIN", "AVG_ARR_DELAY_MIN"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Brakuje kolumn w Silver: {missing}")


date_range = df.agg(F.min("DATE").alias("min_date"), F.max("DATE").alias("max_date")).collect()[0]
print("Zakres dat:", date_range["min_date"], "->", date_range["max_date"])

years = [r["YEAR"] for r in df.select("YEAR").distinct().orderBy("YEAR").collect()]
print("Lata w danych:", years)


overall = df.agg(
    F.count("*").alias("days_count"),
    F.sum("ARR_TOTAL_FLIGHTS").alias("arrivals_sum"),
    F.avg("ARR_TOTAL_FLIGHTS").alias("arrivals_avg_per_day"),
    F.sum("TOTAL_DELAY_MIN").alias("delay_min_sum"),
    F.avg("AVG_ARR_DELAY_MIN").alias("avg_delay_min_mean"),
    F.expr("percentile_approx(AVG_ARR_DELAY_MIN, 0.5)").alias("avg_delay_min_median"),
    F.max("AVG_ARR_DELAY_MIN").alias("avg_delay_min_max")
).withColumn("airport", F.lit("EPWA"))

print("\n OVERALL")
overall.show(truncate=False)

yearly = df.groupBy("YEAR").agg(
    F.count("*").alias("days_count"),
    F.sum("ARR_TOTAL_FLIGHTS").alias("arrivals_sum"),
    F.avg("ARR_TOTAL_FLIGHTS").alias("arrivals_avg_per_day"),
    F.sum("TOTAL_DELAY_MIN").alias("delay_min_sum"),
    F.avg("AVG_ARR_DELAY_MIN").alias("avg_delay_min_mean"),
    F.expr("percentile_approx(AVG_ARR_DELAY_MIN, 0.5)").alias("avg_delay_min_median"),
    F.max("AVG_ARR_DELAY_MIN").alias("avg_delay_min_max")
).orderBy("YEAR").withColumn("airport", F.lit("EPWA"))

print("\n YEARLY")
yearly.show(20, truncate=False)

monthly = df.groupBy("YEAR", "MONTH").agg(
    F.count("*").alias("days_count"),
    F.sum("ARR_TOTAL_FLIGHTS").alias("arrivals_sum"),
    F.avg("ARR_TOTAL_FLIGHTS").alias("arrivals_avg_per_day"),
    F.sum("TOTAL_DELAY_MIN").alias("delay_min_sum"),
    F.avg("AVG_ARR_DELAY_MIN").alias("avg_delay_min_mean"),
    F.expr("percentile_approx(AVG_ARR_DELAY_MIN, 0.5)").alias("avg_delay_min_median"),
    F.max("AVG_ARR_DELAY_MIN").alias("avg_delay_min_max")
).orderBy("YEAR", "MONTH").withColumn("airport", F.lit("EPWA"))

print("\n MONTHLY")
monthly.show(24, truncate=False)


weekday = df.groupBy("DOW").agg(
    F.count("*").alias("days_count"),
    F.avg("ARR_TOTAL_FLIGHTS").alias("arrivals_avg_per_day"),
    F.avg("AVG_ARR_DELAY_MIN").alias("avg_delay_min_mean"),
    F.expr("percentile_approx(AVG_ARR_DELAY_MIN, 0.5)").alias("avg_delay_min_median")
).orderBy("DOW").withColumn("airport", F.lit("EPWA"))

print("\n WEEKDAY")
weekday.show(truncate=False)


cause_cols = [c for c in df.columns if c.startswith("DLY_APT_ARR_") and c.endswith("_1")]

cause_cols = [c for c in cause_cols if c not in ["DLY_APT_ARR_NA_1"]]

if cause_cols:
    
    exprs = [F.sum(F.col(c)).alias(c) for c in cause_cols]
    causes_wide = df.agg(*exprs).withColumn("airport", F.lit("EPWA"))

    
    stack_expr = ", ".join([f"'{c}', {c}" for c in cause_cols])
    causes_long = causes_wide.selectExpr("airport", f"stack({len(cause_cols)}, {stack_expr}) as (cause, minutes)") \
                             .withColumn("minutes", F.col("minutes").cast("double")) \
                             .withColumn("minutes", F.when(F.col("minutes").isNull(), F.lit(0.0)).otherwise(F.col("minutes")))

    total_minutes = causes_long.agg(F.sum("minutes").alias("total")).collect()[0]["total"] or 0.0
    causes_long = causes_long.withColumn(
        "share",
        F.when(F.lit(total_minutes) > 0, F.col("minutes") / F.lit(total_minutes)).otherwise(F.lit(0.0))
    ).orderBy(F.desc("minutes"))

    print("\n CAUSES")
    causes_long.show(20, truncate=False)
else:
    causes_long = None
    print("\nBrak kolumn DLY_APT_ARR_* w Silver – pomijam analizę przyczyn.")


overall_out = f"{GOLD_PREFIX}/overall"
yearly_out  = f"{GOLD_PREFIX}/yearly"
monthly_out = f"{GOLD_PREFIX}/monthly"
weekday_out = f"{GOLD_PREFIX}/weekday"
causes_out  = f"{GOLD_PREFIX}/causes"

overall.write.mode("overwrite").parquet(overall_out)
yearly.write.mode("overwrite").parquet(yearly_out)
monthly.write.mode("overwrite").parquet(monthly_out)
weekday.write.mode("overwrite").parquet(weekday_out)

if causes_long is not None:
    causes_long.write.mode("overwrite").parquet(causes_out)

print("\nZapisano EDA do Gold:")
print(" -", overall_out)
print(" -", yearly_out)
print(" -", monthly_out)
print(" -", weekday_out)
if causes_long is not None:
    print(" -", causes_out)

spark.stop()

