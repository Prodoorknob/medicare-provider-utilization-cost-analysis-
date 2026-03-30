# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver Clean — Null Handling, Type Casting, Outlier Removal
# MAGIC Reads from the Bronze Delta table, applies data quality rules,
# MAGIC and writes a cleaned Silver Delta table ready for feature engineering.

# COMMAND ----------

BRONZE_TABLE = "medicare.bronze_physician_practitioners"
SILVER_TABLE = "medicare.silver_physician_practitioners"

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

df = spark.table(BRONZE_TABLE)

# --- 1. Cast numeric columns from string ---
NUMERIC_COLS = [
    "Tot_Benes", "Tot_Srvcs", "Tot_Bene_Day_Srvcs",
    "Avg_Sbmtd_Chrg", "Avg_Mdcr_Allo_Amt", "Avg_Mdcr_Pymt_Amt",
    "Avg_Mdcr_Stdzd_Amt",
]

for col in NUMERIC_COLS:
    df = df.withColumn(col, F.col(col).cast(DoubleType()))

# --- 2. Drop rows missing the target or key identifiers ---
REQUIRED_COLS = ["Rndrng_NPI", "HCPCS_Cd", "Avg_Mdcr_Pymt_Amt"]
df = df.dropna(subset=REQUIRED_COLS)

# --- 3. Remove statistical outliers on the target (IQR method) ---
q1, q3 = df.approxQuantile("Avg_Mdcr_Pymt_Amt", [0.25, 0.75], 0.01)
iqr = q3 - q1
lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
df = df.filter(F.col("Avg_Mdcr_Pymt_Amt").between(lower, upper))

# --- 4. Normalize free-text provider type ---
df = df.withColumn(
    "Rndrng_Prvdr_Type",
    F.trim(F.initcap(F.col("Rndrng_Prvdr_Type")))
)

print(f"Silver row count after cleaning: {df.count():,}")

# COMMAND ----------

(
    df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(SILVER_TABLE)
)

print(f"Silver table '{SILVER_TABLE}' written successfully.")
