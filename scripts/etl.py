import os
from pyspark.sql import SparkSession, functions as F


BUCKET = "epwa-delays-project"

BRONZE_LOCAL = "/home/ubuntu/epwa/data/bronze/eurocontrol"
SILVER_LOCAL = "/home/ubuntu/epwa/data/silver/daily"


spark = (
    SparkSession.builder
    .appName("EPWA-ETL-CODA")
    .config("spark.sql.legacy.timeParserPolicy", "LEGACY")
    .getOrCreate()
)


df = (
    spark.read
    .option("header", True)
    .option("recursiveFileLookup", "true")
    
    .csv(BRONZE_LOCAL)
)


df = df.withColumn("APT_ICAO", F.upper(F.col("APT_ICAO")))
df = df.withColumn("APT_ICAO", F.regexp_replace("APT_ICAO", r"[^A-Z0-9]", ""))
df = df.filter(F.col("APT_ICAO") == "EPWA")



base_num_cols = ["YEAR", "MONTH_NUM", "FLT_ARR_1", "FLT_ARR_1_DLY", "FLT_ARR_1_DLY_15"]
delay_cols = [
    "DLY_APT_ARR_A_1","DLY_APT_ARR_C_1","DLY_APT_ARR_D_1","DLY_APT_ARR_E_1",
    "DLY_APT_ARR_G_1","DLY_APT_ARR_I_1","DLY_APT_ARR_M_1","DLY_APT_ARR_N_1",
    "DLY_APT_ARR_O_1","DLY_APT_ARR_P_1","DLY_APT_ARR_R_1","DLY_APT_ARR_S_1",
    "DLY_APT_ARR_T_1","DLY_APT_ARR_V_1","DLY_APT_ARR_W_1","DLY_APT_ARR_NA_1",
    "DLY_APT_ARR_1"  # suma minut opóźnień przylotów (jeśli obecna)
]
for c in base_num_cols + delay_cols:
    if c in df.columns:
        df = df.withColumn(c, F.col(c).cast("double"))



ts  = F.to_timestamp("FLT_DATE", "yyyy-MM-dd'T'HH:mm:ssX")
ts2 = F.to_timestamp("FLT_DATE", "yyyy-MM-dd'T'HH:mm:ss.SSSX")  
df  = df.withColumn("DATE_TS", F.coalesce(ts, ts2))
df  = df.withColumn("DATE", F.to_date("DATE_TS"))


df = df.withColumn("YEAR",  F.year("DATE"))
df = df.withColumn("MONTH", F.month("DATE"))


dow = F.dayofweek("DATE")
df = df.withColumn("DOW", ((dow + 5) % 7) + 1)


present_delay_cols = [c for c in delay_cols if c in df.columns and c != "DLY_APT_ARR_1"]

sum_delays_expr = None
for c in present_delay_cols:
    sum_delays_expr = (
        F.coalesce(F.col(c), F.lit(0.0))
        if sum_delays_expr is None
        else (sum_delays_expr + F.coalesce(F.col(c), F.lit(0.0)))
    )


df = df.withColumn(
    "TOTAL_DELAY_MIN",
    F.when(F.col("DLY_APT_ARR_1").isNotNull(), F.col("DLY_APT_ARR_1"))
     .otherwise(sum_delays_expr)
)

df = df.withColumn("ARR_TOTAL_FLIGHTS", F.col("FLT_ARR_1").cast("double"))

df = df.withColumn(
    "AVG_ARR_DELAY_MIN",
    F.when((F.col("ARR_TOTAL_FLIGHTS") > 0) & F.col("TOTAL_DELAY_MIN").isNotNull(),
           F.col("TOTAL_DELAY_MIN") / F.col("ARR_TOTAL_FLIGHTS"))
     .otherwise(F.lit(None))
)


df = df.withColumnRenamed("APT_ICAO", "AIRPORT")

ordered = [
    "AIRPORT","DATE","YEAR","MONTH","DOW",
    "ARR_TOTAL_FLIGHTS","FLT_ARR_1_DLY","FLT_ARR_1_DLY_15",
    "TOTAL_DELAY_MIN","AVG_ARR_DELAY_MIN"
] + [c for c in delay_cols if c in df.columns]

df_out = df.select([c for c in ordered if c in df.columns])


(
    df_out.write
    .mode("overwrite")
    .partitionBy("YEAR","MONTH")
    .parquet(SILVER_LOCAL)
)

print(f" Silver zapisany lokalnie: {SILVER_LOCAL}")


exit_code = os.system(f"aws s3 sync {SILVER_LOCAL} s3://{BUCKET}/silver/daily/")
if exit_code == 0:
    print(f" Silver wysłany do s3://{BUCKET}/silver/daily/")
else:
    print(" Uwaga: problem z wysyłką do S3 ")


rows = df_out.count()
dates = df_out.agg(F.min("DATE").alias("minD"), F.max("DATE").alias("maxD")).collect()[0]
print(f" Rekordów (dni): {rows}, zakres dat: {dates['minD']} → {dates['maxD']}")

