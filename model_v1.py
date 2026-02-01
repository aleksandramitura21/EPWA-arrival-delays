from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row


BUCKET = "epwa-delays-project"

SILVER_PATH = f"s3a://{BUCKET}/silver/daily/"
OUT_BASE    = f"s3a://{BUCKET}/gold/model/v1"

MODEL_PATH      = f"{OUT_BASE}/model"
PRED_PATH       = f"{OUT_BASE}/predictions"        # predykcje tylko dla test_df (metryki)
PRED_FULL_PATH  = f"{OUT_BASE}/predictions_full"   # predykcje dla całego df (wykresy)
FI_PATH         = f"{OUT_BASE}/feature_importance"
METRICS_PATH    = f"{OUT_BASE}/metrics"

AIRPORT_CODE = "EPWA"


RF_NUM_TREES = 200
RF_MAX_DEPTH = 8
SEED = 42



spark = (
    SparkSession.builder
    .appName("MODEL-V1-EPWA")
    .config("spark.sql.shuffle.partitions", "8")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")



df = spark.read.parquet(SILVER_PATH)


if "AIRPORT" in df.columns:
    df = df.filter(F.col("AIRPORT") == F.lit(AIRPORT_CODE))
else:
    raise RuntimeError("Brak kolumny AIRPORT w SILVER")


if "DATE" not in df.columns:
    raise RuntimeError("Brak kolumny DATE w SILVER")

LABEL_COL = "AVG_ARR_DELAY_MIN"
if LABEL_COL not in df.columns:
    raise RuntimeError(f"Brak kolumny label {LABEL_COL} w SILVER. Kolumny: {df.columns}")


if "YEAR" not in df.columns:
    df = df.withColumn("YEAR", F.year(F.col("DATE")))
if "MONTH" not in df.columns:
    df = df.withColumn("MONTH", F.month(F.col("DATE")))


df = (
    df
    .withColumn("DOW", F.dayofweek(F.col("DATE")))    # 1..7
    .withColumn("DOM", F.dayofmonth(F.col("DATE")))   # 1..31
    .withColumn("WOY", F.weekofyear(F.col("DATE")))   # 1..53
)


candidate_features = [
    "ARR_TOTAL_FLIGHTS",
    "FLT_ARR_1_DLY",
    "FLT_ARR_1_DLY_15",

    "DLY_APT_ARR_A_1",
    "DLY_APT_ARR_C_1",
    "DLY_APT_ARR_D_1",
    "DLY_APT_ARR_E_1",
    "DLY_APT_ARR_G_1",
    "DLY_APT_ARR_I_1",
    "DLY_APT_ARR_M_1",
    "DLY_APT_ARR_N_1",
    "DLY_APT_ARR_O_1",
    "DLY_APT_ARR_P_1",
    "DLY_APT_ARR_R_1",
    "DLY_APT_ARR_S_1",
    "DLY_APT_ARR_T_1",
    "DLY_APT_ARR_V_1",
    "DLY_APT_ARR_W_1",
    "DLY_APT_ARR_NA_1",
    "DLY_APT_ARR_1",

    "YEAR", "MONTH", "DOW", "DOM", "WOY",
]

feature_cols = [c for c in candidate_features if c in df.columns]

if len(feature_cols) == 0:
    raise RuntimeError("Nie znaleziono żadnych feature columns w df. Sprawdź schemat SILVER.")


for c in feature_cols:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.fillna(0.0, subset=feature_cols)


df = df.withColumn(LABEL_COL, F.col(LABEL_COL).cast("double"))


df = df.dropna(subset=["DATE", LABEL_COL])


df = df.dropDuplicates(["DATE"])

print("\n FEATURES USED")
for c in feature_cols:
    print(" -", c)

=
years = [r["YEAR"] for r in df.select("YEAR").distinct().collect()]
years_sorted = sorted([int(y) for y in years]) if years else []

if len(years_sorted) >= 2:
    test_year = years_sorted[-1]  # ostatni rok jako test
    train_df = df.filter(F.col("YEAR") < F.lit(test_year))
    test_df  = df.filter(F.col("YEAR") == F.lit(test_year))

    if test_df.count() < 30 or train_df.count() < 100:
        train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)
        split_mode = "RANDOM_SPLIT_80_20"
    else:
        split_mode = f"TIME_SPLIT_TEST_YEAR_{test_year}"
else:
    train_df, test_df = df.randomSplit([0.8, 0.2], seed=SEED)
    split_mode = "RANDOM_SPLIT_80_20"

print("\n SPLIT MODE")
print(split_mode)
print("train rows:", train_df.count())
print("test rows :", test_df.count())


assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

rf = RandomForestRegressor(
    featuresCol="features",
    labelCol=LABEL_COL,
    predictionCol="prediction",
    numTrees=RF_NUM_TREES,
    maxDepth=RF_MAX_DEPTH,
    seed=SEED
)

pipeline = Pipeline(stages=[assembler, rf])


model = pipeline.fit(train_df)


test_pred = model.transform(test_df)

pred_out = test_pred.select(
    F.col("DATE").alias("DATE"),
    F.col("YEAR").cast("int").alias("YEAR"),
    F.col("MONTH").cast("int").alias("MONTH"),
    F.col(LABEL_COL).alias("AVG_ARR_DELAY_MIN"),
    F.col("prediction").alias("prediction")
)


full_pred = model.transform(df)

pred_full_out = full_pred.select(
    F.col("DATE").alias("DATE"),
    F.col("YEAR").cast("int").alias("YEAR"),
    F.col("MONTH").cast("int").alias("MONTH"),
    F.col(LABEL_COL).alias("AVG_ARR_DELAY_MIN"),
    F.col("prediction").alias("prediction")
)

rmse = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="rmse"
).evaluate(test_pred)

r2 = RegressionEvaluator(
    labelCol=LABEL_COL,
    predictionCol="prediction",
    metricName="r2"
).evaluate(test_pred)

print("\n METRICS (TEST)")
print("RMSE:", rmse)
print("R2  :", r2)


metrics_df = spark.createDataFrame([
    Row(metric="RMSE", value=float(rmse), info=""),
    Row(metric="R2", value=float(r2), info=""),
    Row(metric="SPLIT_MODE", value=0.0, info=split_mode),
])


rf_model = model.stages[-1]
importances = rf_model.featureImportances.toArray().tolist()

fi_rows = [
    Row(feature=feature_cols[i], importance=float(importances[i]))
    for i in range(len(feature_cols))
]
fi_df = spark.createDataFrame(fi_rows).orderBy(F.col("importance").desc())

print("\n TOP 20 FEATURE IMPORTANCE")
fi_df.show(20, False)


model.write().overwrite().save(MODEL_PATH)


pred_out.write.mode("overwrite").parquet(PRED_PATH)


pred_full_out.write.mode("overwrite").parquet(PRED_FULL_PATH)


metrics_df.write.mode("overwrite").parquet(METRICS_PATH)


fi_df.write.mode("overwrite").parquet(FI_PATH)

spark.stop()
p
