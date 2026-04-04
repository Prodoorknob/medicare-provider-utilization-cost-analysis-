# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Exploratory Data Analysis
# MAGIC Distribution plots, correlation heatmap, and provider-level summaries
# MAGIC using the Silver Delta table.

# COMMAND ----------

SILVER_TABLE = "medicare.silver_physician_practitioners"

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------
# MAGIC %md ## 1 · Target Distribution — Avg Medicare Payment

df_pd = spark.table(SILVER_TABLE).select(
    "Avg_Mdcr_Pymt_Amt", "Avg_Sbmtd_Chrg", "Avg_Mdcr_Alowd_Amt",
    "Tot_Srvcs", "Tot_Benes", "Rndrng_Prvdr_Type", "Rndrng_Prvdr_State_Abrvtn"
).sample(fraction=0.05, seed=42).toPandas()

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, ["Avg_Mdcr_Pymt_Amt", "Avg_Sbmtd_Chrg", "Tot_Srvcs"]):
    df_pd[col].dropna().plot.hist(bins=60, ax=ax, title=col)
    ax.set_xlabel(col)
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md ## 2 · Correlation Heatmap

NUM_COLS = [
    "Tot_Benes", "Tot_Srvcs", "Avg_Sbmtd_Chrg",
    "Avg_Mdcr_Alowd_Amt", "Avg_Mdcr_Pymt_Amt", "Avg_Mdcr_Stdzd_Amt"
]
corr = df_pd[NUM_COLS].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix — Medicare Cost Features")
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md ## 3 · Top Provider Types by Avg Payment

top_types = (
    df_pd.groupby("Rndrng_Prvdr_Type")["Avg_Mdcr_Pymt_Amt"]
    .median()
    .nlargest(20)
    .reset_index()
)
top_types.columns = ["Provider Type", "Median Avg Payment"]

plt.figure(figsize=(10, 6))
sns.barplot(data=top_types, x="Median Avg Payment", y="Provider Type")
plt.title("Top 20 Provider Types by Median Avg Medicare Payment")
plt.tight_layout()
plt.show()

# COMMAND ----------
# MAGIC %md ## 4 · State-Level Average Payment Map (summary table)

state_summary = (
    df_pd.groupby("Rndrng_Prvdr_State_Abrvtn")["Avg_Mdcr_Pymt_Amt"]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
    .reset_index()
)
state_summary.columns = ["State", "Mean Payment", "Median Payment", "Records"]
display(state_summary)
