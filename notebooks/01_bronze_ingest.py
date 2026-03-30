# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze Ingest — S3 → Delta Table
# MAGIC Reads raw Medicare physician/practitioner CSVs from S3 and writes
# MAGIC them as-is into the Bronze Delta table. No transformations here.

# COMMAND ----------

# Configuration — override via Databricks widgets or job parameters
S3_RAW_PATH   = "s3://your-bucket/medicare/raw/*.csv"
BRONZE_TABLE  = "medicare.bronze_physician_practitioners"
CATALOG       = "hive_metastore"   # or Unity Catalog name

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Read all yearly CSVs from S3 with schema inference disabled (preserve raw strings)
df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")   # keep all columns as string in Bronze
    .option("mergeSchema", "true")    # tolerate minor schema drift across years
    .csv(S3_RAW_PATH)
)

print(f"Rows ingested: {df_raw.count():,}")
df_raw.printSchema()

# COMMAND ----------

# Write to Delta — overwrite for full reload; switch to append + dedup for incremental
(
    df_raw.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(BRONZE_TABLE)
)

print(f"Bronze table '{BRONZE_TABLE}' written successfully.")
