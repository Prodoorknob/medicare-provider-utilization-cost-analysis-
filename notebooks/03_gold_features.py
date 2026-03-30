# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold Features — Feature Engineering & Export
# MAGIC Reads the Silver table, engineers model-ready features, and writes:
# MAGIC - A Gold Delta table for downstream Databricks jobs
# MAGIC - A Parquet export consumed by local modeling scripts

# COMMAND ----------

SILVER_TABLE  = "medicare.silver_physician_practitioners"
GOLD_TABLE    = "medicare.gold_features"
GOLD_EXPORT   = "s3://your-bucket/medicare/gold/features.parquet"  # or DBFS path

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

df = spark.table(SILVER_TABLE)

# --- 1. Derived ratio features ---
df = (
    df
    .withColumn("srvcs_per_bene",
                F.col("Tot_Srvcs") / F.when(F.col("Tot_Benes") > 0, F.col("Tot_Benes")).otherwise(None))
    .withColumn("pymt_to_charge_ratio",
                F.col("Avg_Mdcr_Pymt_Amt") / F.when(F.col("Avg_Sbmtd_Chrg") > 0, F.col("Avg_Sbmtd_Chrg")).otherwise(None))
    .withColumn("stdz_to_pymt_ratio",
                F.col("Avg_Mdcr_Stdzd_Amt") / F.when(F.col("Avg_Mdcr_Pymt_Amt") > 0, F.col("Avg_Mdcr_Pymt_Amt")).otherwise(None))
)

# --- 2. Encode high-cardinality categoricals ---
CAT_COLS = ["Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn", "HCPCS_Cd"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in CAT_COLS]
pipeline  = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

# --- 3. Select final feature set ---
FEATURE_COLS = [
    "Rndrng_Prvdr_Type_idx", "Rndrng_Prvdr_State_Abrvtn_idx", "HCPCS_Cd_idx",
    "Tot_Benes", "Tot_Srvcs",
    "Avg_Sbmtd_Chrg", "Avg_Mdcr_Allo_Amt", "Avg_Mdcr_Stdzd_Amt",
    "srvcs_per_bene", "pymt_to_charge_ratio", "stdz_to_pymt_ratio",
    "Avg_Mdcr_Pymt_Amt",   # target
]
df_gold = df.select(*FEATURE_COLS).dropna()

print(f"Gold feature row count: {df_gold.count():,}")
df_gold.printSchema()

# COMMAND ----------

df_gold.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_TABLE)
df_gold.write.mode("overwrite").parquet(GOLD_EXPORT)

print(f"Gold table '{GOLD_TABLE}' and Parquet export written.")
