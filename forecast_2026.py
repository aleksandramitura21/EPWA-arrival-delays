import os
from datetime import datetime

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import PipelineModel


BUCKET = "epwa-delays-project"
AIRPORT_CODE = "EPWA"

SILVER_PATH = f"s3a://{BUCKET}/silver/daily/"
MODEL_PATH  = f"s3a://{BUCKET}/gold/model/v1/model"

OUT_DAILY   = f"s3a://{BUCKET}/gold/forecast/2026/daily"
OUT_MONTHLY = f"s3a://{BUCKET}/gold/forecast/2026/monthly"

S3_PLOTS_PREFIX = "gold/plots/forecast_2026"
LOCAL_PLOTS_DIR = os.path.expanduser("~/epwa/raport/plots")

HIST_START_YEAR = 2022
HIST_END_YEAR = 2025


spark = (
    SparkSession.builder
    .appName("FORECAST-2026-EPWA")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")




df = spark.read.parquet(SILVER_PATH)

if "AIRPORT" in df.columns:
    df = df.filter(F.col("AIRPORT") == F.lit(AIRPORT_CODE))


if "DATE" not in df.columns:
    spark.stop()
    raise RuntimeError("Brak kolumny DATE w SILVER.")


if "YEAR" not in df.columns:
    df = df.withColumn("YEAR", F.year("DATE"))
if "MONTH" not in df.columns:
    df = df.withColumn("MONTH", F.month("DATE"))

df = (
    df.withColumn("DOW", F.dayofweek("DATE"))      # 1=Sun..7=Sat
      .withColumn("DOM", F.dayofmonth("DATE"))     # 1..31
      .withColumn("WOY", F.weekofyear("DATE"))     # 1..53
)

df_hist = df.filter((F.col("YEAR") >= HIST_START_YEAR) & (F.col("YEAR") <= HIST_END_YEAR))


 2) Załaduj model i pobierz inputCols

model = PipelineModel.load(MODEL_PATH)


assembler = model.stages[0]
input_cols = assembler.getInputCols()

print("\n MODEL INPUT COLS")
print(input_cols)


calendar_cols = {"YEAR", "MONTH", "DOW", "DOM", "WOY"}


non_calendar_cols = [c for c in input_cols if c not in calendar_cols]

print("\nNON-CALENDAR COLS (baseline from 2022–2025)")
print(non_calendar_cols)



need_for_baseline = ["MONTH", "DOW"] + non_calendar_cols
existing_cols = [c for c in need_for_baseline if c in df_hist.columns]

missing_in_silver = [c for c in non_calendar_cols if c not in df_hist.columns]
if missing_in_silver:
    print("\n Te kolumny są w modelu, ale nie ma ich w SILVER")
    print(missing_in_silver)

df_base = df_hist.select(*existing_cols)


for c in non_calendar_cols:
    if c in df_base.columns:
        df_base = df_base.withColumn(c, F.col(c).cast("double"))
df_base = df_base.fillna(0.0)


agg_exprs = []
for c in non_calendar_cols:
    if c in df_base.columns:
        agg_exprs.append(F.avg(F.col(c)).alias(c))

baseline_md = df_base.groupBy("MONTH", "DOW").agg(*agg_exprs)


global_means_row = df_base.agg(*[F.avg(F.col(c)).alias(c) for c in non_calendar_cols if c in df_base.columns]).collect()[0].asDict()
global_means = {k: float(v) if v is not None else 0.0 for k, v in global_means_row.items()}


for c in missing_in_silver:
    global_means[c] = 0.0

print("\nGLOBAL MEANS (fallback)")
for k in sorted(global_means.keys()):
    print(k, "=", global_means[k])




dates_2026 = (
    spark.sql("SELECT sequence(date('2026-01-01'), date('2026-12-31'), interval 1 day) as d")
         .select(F.explode("d").alias("DATE"))
         .withColumn("YEAR", F.lit(2026).cast("int"))
         .withColumn("MONTH", F.month("DATE").cast("int"))
         .withColumn("DOW", F.dayofweek("DATE").cast("int"))
         .withColumn("DOM", F.dayofmonth("DATE").cast("int"))
         .withColumn("WOY", F.weekofyear("DATE").cast("int"))
)




future = dates_2026.join(baseline_md, on=["MONTH", "DOW"], how="left")


for c in non_calendar_cols:
    if c in future.columns:
        future = future.withColumn(c, F.when(F.col(c).isNull(), F.lit(global_means.get(c, 0.0))).otherwise(F.col(c)))
    else:
        
        future = future.withColumn(c, F.lit(global_means.get(c, 0.0)).cast("double"))


for c in input_cols:
    if c in calendar_cols:
        future = future.withColumn(c, F.col(c).cast("int"))
    else:
        future = future.withColumn(c, F.col(c).cast("double"))




pred = model.transform(future)


out_daily = pred.select(
    F.col("DATE"),
    F.col("YEAR").cast("int").alias("YEAR"),
    F.col("MONTH").cast("int").alias("MONTH"),
    F.col("DOW").cast("int").alias("DOW"),
    F.col("prediction").cast("double").alias("pred_avg_arr_delay_min")
).orderBy("DATE")


out_monthly = (
    out_daily.groupBy("YEAR", "MONTH")
             .agg(
                 F.count("*").alias("days_count"),
                 F.avg("pred_avg_arr_delay_min").alias("pred_monthly_mean_delay_min")
             )
             .orderBy("YEAR", "MONTH")
)



out_daily.write.mode("overwrite").parquet(OUT_DAILY)
out_monthly.write.mode("overwrite").parquet(OUT_MONTHLY)

print(f"\n[OK] Daily forecast saved:   {OUT_DAILY}")
print(f"[OK] Monthly forecast saved: {OUT_MONTHLY}")

os.makedirs(LOCAL_PLOTS_DIR, exist_ok=True)
ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

pdf = out_monthly.toPandas()
pdf["year_month"] = pdf["YEAR"].astype(str) + "-" + pdf["MONTH"].astype(str).str.zfill(2)

fig = plt.figure(figsize=(14, 6))
plt.plot(pdf["year_month"], pdf["pred_monthly_mean_delay_min"], marker="o", label="Forecast 2026 (monthly mean)")
plt.title("Prognoza opóźnień ATFM (przyloty) — EPWA, rok 2026 (scenariusz 2022–2025)")
plt.xlabel("Rok-Miesiąc")
plt.ylabel("Predicted avg delay [min]")
plt.grid(True, axis="y", linewidth=0.5)
plt.legend()


n = len(pdf)
step = max(1, n // 12)  
ticks = list(range(0, n, step))
plt.xticks(ticks, [pdf["year_month"].iloc[i] for i in ticks], rotation=45, ha="right")

plt.tight_layout()

filename = f"forecast_2026_monthly_{ts}.png"
local_path = os.path.join(LOCAL_PLOTS_DIR, filename)
fig.savefig(local_path, dpi=160, bbox_inches="tight")
plt.close(fig)


s3 = boto3.client("s3")
s3_key = f"{S3_PLOTS_PREFIX}/{filename}"
s3.upload_file(local_path, BUCKET, s3_key)

print(f"[OK] PNG uploaded: s3://{BUCKET}/{s3_key}")

spark.stop()

