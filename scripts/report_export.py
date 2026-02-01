import os
from datetime import datetime

from pyspark.sql import SparkSession, functions as F


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BUCKET = "epwa-delays-project"
AIRPORT = "EPWA"

SILVER_PATH = f"s3a://{BUCKET}/silver/daily/"
CAUSES_PATH = f"s3a://{BUCKET}/gold/eda/causes/"
ANOM_PATH   = f"s3a://{BUCKET}/gold/eda/anomalies/"
VALM_PATH   = f"s3a://{BUCKET}/gold/model/v1/validation_monthly/"
INTERP_PATH = f"s3a://{BUCKET}/gold/model/v1/interpretation/"

OUT_BASE = os.path.expanduser("~/epwa/raport")
OUT_TABLES = os.path.join(OUT_BASE, "tables")
OUT_FIGS   = os.path.join(OUT_BASE, "figures")
OUT_LOGS   = os.path.join(OUT_BASE, "logs")


def ensure_dirs():
    os.makedirs(OUT_TABLES, exist_ok=True)
    os.makedirs(OUT_FIGS, exist_ok=True)
    os.makedirs(OUT_LOGS, exist_ok=True)


def save_csv(df_spark, out_csv_path, order_by=None, limit=None):
    d = df_spark
    if order_by is not None:
        d = d.orderBy(*order_by)
    if limit is not None:
        d = d.limit(limit)
    pdf = d.toPandas()
    pdf.to_csv(out_csv_path, index=False)
    return pdf


def plot_line(pdf, x_cols, y_cols, title, out_png, xlabel=None, ylabel=None):
    plt.figure()
    if isinstance(x_cols, list) and len(x_cols) > 1:
        x = pdf[x_cols].astype(str).agg("-".join, axis=1)
    else:
        xcol = x_cols[0] if isinstance(x_cols, list) else x_cols
        x = pdf[xcol].astype(str)

    for y in y_cols:
        plt.plot(x, pdf[y])

    plt.title(title)
    plt.xticks(rotation=60, ha="right")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_bar(pdf, x_col, y_col, title, out_png, xlabel=None, ylabel=None, top_n=None):
    plt.figure()
    d = pdf.copy()
    if top_n is not None:
        d = d.head(top_n)
    plt.bar(d[x_col].astype(str), d[y_col])
    plt.title(title)
    plt.xticks(rotation=60, ha="right")
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ensure_dirs()

    spark = (
        SparkSession.builder
        .appName("EPWA-REPORT-EXPORT")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    silver = spark.read.parquet(SILVER_PATH)

    if "AIRPORT" in silver.columns:
        silver = silver.filter(F.col("AIRPORT") == AIRPORT)

    silver_basic = silver.select(
        F.col("DATE").alias("date"),
        F.col("YEAR").cast("int").alias("year"),
        F.col("MONTH").cast("int").alias("month"),
        F.col("AVG_ARR_DELAY_MIN").cast("double").alias("avg_arr_delay_min"),
    ).dropna(subset=["date", "year", "month"])

    t_data_range = silver_basic.agg(
        F.min("date").alias("min_date"),
        F.max("date").alias("max_date"),
        F.count("*").alias("days_count"),
        F.countDistinct("year").alias("years_count"),
    )
    save_csv(t_data_range, os.path.join(OUT_TABLES, f"data_range_{stamp}.csv"))

    t_days_per_year = (silver_basic
        .groupBy("year")
        .agg(F.count("*").alias("days_count"))
        .orderBy("year")
    )
    pdf_days = save_csv(t_days_per_year, os.path.join(OUT_TABLES, f"days_per_year_{stamp}.csv"))

    plot_bar(
        pdf_days, "year", "days_count",
        title="Liczba dni danych w latach (EPWA)",
        out_png=os.path.join(OUT_FIGS, f"days_per_year_{stamp}.png"),
        xlabel="Rok", ylabel="Liczba dni"
    )

    t_monthly_delay = (silver_basic
        .groupBy("year", "month")
        .agg(F.avg("avg_arr_delay_min").alias("monthly_avg_delay_min"))
        .orderBy("year", "month")
    )
    pdf_monthly = save_csv(t_monthly_delay, os.path.join(OUT_TABLES, f"silver_monthly_delay_{stamp}.csv"))

    plot_line(
        pdf_monthly, ["year", "month"], ["monthly_avg_delay_min"],
        title="Średnie opóźnienie przylotów ATFM (miesięcznie) — EPWA",
        out_png=os.path.join(OUT_FIGS, f"silver_monthly_delay_{stamp}.png"),
        xlabel="Rok-Miesiąc", ylabel="Avg delay [min]"
    )

    causes = spark.read.parquet(CAUSES_PATH)
    if "airport" in causes.columns:
        causes = causes.filter(F.col("airport") == AIRPORT)
    elif "AIRPORT" in causes.columns:
        causes = causes.filter(F.col("AIRPORT") == AIRPORT)

    c = causes.select(
        F.col("cause").cast("string").alias("cause"),
        F.col("minutes").cast("double").alias("minutes"),
        F.col("share").cast("double").alias("share"),
    ).dropna(subset=["cause"])

    t_causes_rank = (c
        .groupBy("cause")
        .agg(F.sum("minutes").alias("total_minutes"),
             F.avg("share").alias("avg_share"))
        .orderBy(F.col("total_minutes").desc())
    )
    pdf_causes = save_csv(t_causes_rank, os.path.join(OUT_TABLES, f"causes_rank_{stamp}.csv"))

    plot_bar(
        pdf_causes, "cause", "total_minutes",
        title="Przyczyny opóźnień ATFM — suma minut (EPWA)",
        out_png=os.path.join(OUT_FIGS, f"causes_total_minutes_{stamp}.png"),
        xlabel="Cause", ylabel="Total minutes", top_n=12
    )

    anomalies = spark.read.parquet(ANOM_PATH)
    if "AIRPORT" in anomalies.columns:
        anomalies = anomalies.filter(F.col("AIRPORT") == AIRPORT)
    elif "airport" in anomalies.columns:
        anomalies = anomalies.filter(F.col("airport") == AIRPORT)

    a = anomalies.select(
        F.col("DATE").alias("date"),
        F.col("YEAR").cast("int").alias("year"),
        F.col("MONTH").cast("int").alias("month"),
        F.col("AVG_ARR_DELAY_MIN").cast("double").alias("avg_arr_delay_min"),
        F.col("anomaly_type").cast("string").alias("anomaly_type"),
        F.col("severity").cast("string").alias("severity"),
    )

    t_anom_top = a.orderBy(F.col("avg_arr_delay_min").desc())
    pdf_anom_top = save_csv(t_anom_top, os.path.join(OUT_TABLES, f"anomalies_top30_{stamp}.csv"), limit=30)

    t_anom_month = (a
        .groupBy("year", "month")
        .agg(F.count("*").alias("anomalies_count"),
             F.avg("avg_arr_delay_min").alias("avg_delay_on_anomaly_days"))
        .orderBy("year", "month")
    )
    pdf_anom_month = save_csv(t_anom_month, os.path.join(OUT_TABLES, f"anomalies_monthly_{stamp}.csv"))

    plot_line(
        pdf_anom_month, ["year", "month"], ["anomalies_count"],
        title="Liczba dni anomalii (miesięcznie) — EPWA",
        out_png=os.path.join(OUT_FIGS, f"anomalies_count_monthly_{stamp}.png"),
        xlabel="Rok-Miesiąc", ylabel="Anomalies count"
    )

    valm = spark.read.parquet(VALM_PATH)

    v = valm.select(
        F.col("year").cast("int").alias("year"),
        F.col("month").cast("int").alias("month"),
        F.col("days_count").cast("int").alias("days_count"),
        F.col("real_avg_delay_mean").cast("double").alias("real_avg_delay_mean"),
        F.col("pred_avg_delay_mean").cast("double").alias("pred_avg_delay_mean"),
        F.col("mae_mean").cast("double").alias("mae_mean"),
        F.col("rmse").cast("double").alias("rmse"),
    )

    pdf_valm = save_csv(v, os.path.join(OUT_TABLES, f"model_validation_monthly_{stamp}.csv"),
                        order_by=["year", "month"])

    plot_line(
        pdf_valm, ["year", "month"], ["real_avg_delay_mean", "pred_avg_delay_mean"],
        title="Walidacja miesięczna: Real vs Pred (EPWA)",
        out_png=os.path.join(OUT_FIGS, f"validation_real_vs_pred_{stamp}.png"),
        xlabel="Rok-Miesiąc", ylabel="Avg delay [min]"
    )

    plot_line(
        pdf_valm, ["year", "month"], ["mae_mean", "rmse"],
        title="Walidacja miesięczna: MAE i RMSE (EPWA)",
        out_png=os.path.join(OUT_FIGS, f"validation_mae_rmse_{stamp}.png"),
        xlabel="Rok-Miesiąc", ylabel="Error [min]"
    )

    interp = spark.read.parquet(INTERP_PATH)

    fi = interp.select(
        F.col("feature").cast("string").alias("feature"),
        F.col("importance").cast("double").alias("importance"),
        F.col("importance_pct").cast("double").alias("importance_pct"),
    ).orderBy(F.col("importance_pct").desc())

    pdf_fi = save_csv(fi, os.path.join(OUT_TABLES, f"model_feature_importance_pct_{stamp}.csv"), limit=200)

    plot_bar(
        pdf_fi, "feature", "importance_pct",
        title="Top cechy modelu (importance_pct) — EPWA",
        out_png=os.path.join(OUT_FIGS, f"feature_importance_top15_{stamp}.png"),
        xlabel="Feature", ylabel="Importance [%]", top_n=15
    )

    with open(os.path.join(OUT_LOGS, f"report_export_{stamp}.txt"), "w", encoding="utf-8") as f:
        f.write(f"Generated at: {stamp}\n")
        f.write(f"OUT_BASE: {OUT_BASE}\n")
        f.write("Tables and figures created.\n")

    spark.stop()


if __name__ == "__main__":
    main()
