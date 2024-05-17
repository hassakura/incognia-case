# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Incognia - Case 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Creating the tables

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Fraud feedback

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/Base_1___Hélio_Assakura___incognia_case_transactions_fraud_feedback.csv"
file_type = "csv"

infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

permanent_table_name = "transactions_fraud_feedback"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM transactions_fraud_feedback

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Transactions

# COMMAND ----------

from pyspark.sql.types import *

customSchema = StructType([
    StructField("transaction_id", StringType(), True),
    StructField("transaction_timestamp", LongType(), True),
    StructField("account_id", StringType(), True),
    StructField("device_id", StringType(), True),
    StructField("distance_to_frequent_location", DoubleType(), True),
    StructField("device_age_days", IntegerType(), True),
    StructField("is_emulator", BooleanType(), True),
    StructField("has_fake_location", BooleanType(), True),
    StructField("has_root_permissions", BooleanType(), True),
    StructField("app_is_tampered", BooleanType(), True),
    StructField("transaction_value", DoubleType(), True),
    StructField("client_decision", StringType(), True)
  ])

# COMMAND ----------

file_location = "/FileStore/tables/Base_2___Hélio_Assakura___incognia_case_transactions.csv"
file_type = "csv"

infer_schema = "false"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .schema(customSchema) \
  .load(file_location)

permanent_table_name = "transactions"

df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM transactions

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Imports

# COMMAND ----------

import pandas as pd
from pyspark.sql.functions import *

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Descriptive analysis of transaction events

# COMMAND ----------


# df_p
df_transactions_raw = spark.read.parquet("dbfs:/user/hive/warehouse/transactions")
df_transactions_feedback_raw = spark.read.parquet("dbfs:/user/hive/warehouse/transactions_fraud_feedback") \
    .withColumn("is_fraud", lit(1))
df_transactions_with_feedback_raw = df_transactions_raw.join(df_transactions_feedback_raw, on = "transaction_id", how = "left")

# COMMAND ----------

df_transactions_with_feedback = df_transactions_with_feedback_raw.select("*").toPandas()

df_transactions_with_feedback["is_emulator"] = df_transactions_with_feedback["is_emulator"].astype('float').astype('Int64')
df_transactions_with_feedback["has_fake_location"] = df_transactions_with_feedback["has_fake_location"].astype('float').astype('Int64')
df_transactions_with_feedback["has_root_permissions"] = df_transactions_with_feedback["has_root_permissions"].astype('float').astype('Int64')
df_transactions_with_feedback["app_is_tampered"] = df_transactions_with_feedback["app_is_tampered"].astype('float').astype('Int64')
df_transactions_with_feedback["client_decision"] = df_transactions_with_feedback["client_decision"].map({'approved':1,'denied':0})
df_transactions_with_feedback["is_fraud"] = df_transactions_with_feedback["is_fraud"].fillna(0)
df_transactions_with_feedback["transcation_timestamp_formatted"] = pd.to_datetime(df_transactions_with_feedback['transaction_timestamp'], unit='ms')

# COMMAND ----------

# MAGIC %md
# MAGIC ### About the data

# COMMAND ----------

pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_transactions_with_feedback.describe()

# COMMAND ----------

df_plot = df_transactions_with_feedback[["transaction_value", "distance_to_frequent_location", "device_age_days"]]
df_plot.plot(
    kind='box', 
    subplots=True, 
    sharey=False, 
    figsize=(20, 9)
)
plt.subplots_adjust(wspace=0.5) 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Attribute's deep dive

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Quartiles

# COMMAND ----------

df_transactions_with_feedback_clean = df_transactions_with_feedback[["distance_to_frequent_location", "device_age_days", "transaction_value"]].dropna()
df_transactions_with_feedback_clean.quantile([0.01, 0.25, 0.5, 0.75, 0.9, 0.99])

# COMMAND ----------

df_transactions_with_feedback_clean.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Transaction Value

# COMMAND ----------

data_zscore_transaction_value = df_transactions_with_feedback[np.abs(stats.zscore(df_transactions_with_feedback["transaction_value"])) < 3]
plt.figure(figsize=(10, 6))
sns.histplot(data=data_zscore_transaction_value, x='transaction_value')
plt.title('Transaction Value')
plt.xlabel('Value (R$)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Distance to Frequent location

# COMMAND ----------

df_transactions_with_feedback_clean = df_transactions_with_feedback[["distance_to_frequent_location", "device_age_days", "transaction_value"]].dropna()
data_zscore_distance = df_transactions_with_feedback_clean[np.abs(stats.zscore(df_transactions_with_feedback_clean["distance_to_frequent_location"])) < 3]
data_zscore_distance = data_zscore_distance.where(data_zscore_distance["distance_to_frequent_location"] < 100)
plt.hist(data_zscore_distance["distance_to_frequent_location"], density=False, bins = 500)
plt.ylabel('Frequency')
plt.xlabel('Distance to frequent location');

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Device age days

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   device_age_days,
# MAGIC   count(DISTINCT transaction_id) AS total_transactions
# MAGIC FROM transactions
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Transaction Time

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC with formatted_timestamp as(
# MAGIC   select
# MAGIC     t.transaction_id,
# MAGIC     timestamp_millis(cast(transaction_timestamp as bigint))  as transcation_timestamp_formatted
# MAGIC   from
# MAGIC     transactions t
# MAGIC )
# MAGIC
# MAGIC
# MAGIC select
# MAGIC 	hour(transcation_timestamp_formatted) as ts_hour,
# MAGIC 	count(formatted_timestamp.transaction_id) as total_transactions
# MAGIC from formatted_timestamp
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Client Decision

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   client_decision,
# MAGIC   count(transaction_id) as total_transactions
# MAGIC FROM transactions
# MAGIC group by 1

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ####  is_emulator, has_fake_location, has_root_permission and app_is_tampered

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select
# MAGIC   'is_emulator' as column_name,
# MAGIC   count(case when is_emulator = true then transaction_id end) as total_true,
# MAGIC   count(case when is_emulator is null then transaction_id end) as total_nulls
# MAGIC from transactions
# MAGIC where is_emulator = true OR is_emulator is null
# MAGIC group by 1
# MAGIC
# MAGIC union all
# MAGIC
# MAGIC select
# MAGIC   'has_fake_location' as column_name,
# MAGIC   count(case when has_fake_location = true then transaction_id end) as total_true,
# MAGIC   count(case when has_fake_location is null then transaction_id end) as total_nulls
# MAGIC from transactions
# MAGIC where has_fake_location = true OR has_fake_location is null
# MAGIC group by 1
# MAGIC
# MAGIC union all
# MAGIC
# MAGIC select
# MAGIC   'has_root_permissions' as column_name,
# MAGIC   count(case when has_root_permissions = true then transaction_id end) as total_true,
# MAGIC   count(case when has_root_permissions is null then transaction_id end) as total_nulls
# MAGIC from transactions
# MAGIC where has_root_permissions = true OR has_root_permissions is null
# MAGIC group by 1
# MAGIC
# MAGIC union all
# MAGIC
# MAGIC select
# MAGIC   'app_is_tampered' as column_name,
# MAGIC   count(case when app_is_tampered = true then transaction_id end) as total_true,
# MAGIC   count(case when app_is_tampered is null then transaction_id end) as total_nulls
# MAGIC from transactions
# MAGIC where app_is_tampered = true OR app_is_tampered is null
# MAGIC group by 1

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### How’s the data quality? How can we fix the problems? Can I trust it?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Null values

# COMMAND ----------

df_transactions_with_feedback.isnull().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Is there a strong correlation between the variables?

# COMMAND ----------

correlation_matrix = df_transactions_with_feedback.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Are there other possible variables that we can create?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Devices per account

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   account_id,
# MAGIC   count(distinct device_id) as total_devices
# MAGIC FROM transactions
# MAGIC   group by 1
# MAGIC   order by 2 desc

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Accounts per Device

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   device_id,
# MAGIC   count(distinct account_id) as total_accounts
# MAGIC FROM
# MAGIC   transactions
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Fraud Risk Classification Data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Fraud data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Fraud Transactions

# COMMAND ----------

df_plot_fraud = df_transactions_with_feedback[["distance_to_frequent_location", "device_age_days", "transaction_value", "is_fraud", "client_decision"]]
df_plot_fraud = df_plot_fraud[df_plot_fraud["is_fraud"] == 1][["transaction_value", "distance_to_frequent_location", "device_age_days"]]
df_plot_fraud.plot(
    kind='box', 
    subplots=True, 
    sharey=False, 
    figsize=(20, 9)
)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Good transactions

# COMMAND ----------

df_plot_no_fraud = df_transactions_with_feedback[["distance_to_frequent_location", "device_age_days", "transaction_value", "is_fraud", "client_decision"]]
df_plot_no_fraud = df_plot_no_fraud[df_plot_no_fraud["is_fraud"] == 0][["transaction_value", "distance_to_frequent_location", "device_age_days"]]
df_plot_no_fraud.plot(
    kind='box', 
    subplots=True, 
    sharey=False, 
    figsize=(20, 9)
)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### New Decision flow: Criteria and Threshold

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### High risk band table creation

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE high_risk_band as(
# MAGIC   with has_more_than_4_accounts_per_device as(
# MAGIC     with timestamp_formated as (
# MAGIC       select
# MAGIC         *,
# MAGIC         timestamp_millis(cast(transaction_timestamp as bigint)) as transcation_timestamp_formatted
# MAGIC       from
# MAGIC         transactions
# MAGIC     ),
# MAGIC     devices_ordered_transaction_history as (
# MAGIC       SELECT
# MAGIC         a.account_id,
# MAGIC         a.transaction_id,
# MAGIC         a.transcation_timestamp_formatted,
# MAGIC         a.device_id,
# MAGIC         count(distinct b.account_id) as total_accounts
# MAGIC       FROM
# MAGIC         timestamp_formated a
# MAGIC         INNER JOIN timestamp_formated b ON a.device_id = b.device_id
# MAGIC         and b.transcation_timestamp_formatted <= a.transcation_timestamp_formatted
# MAGIC       GROUP BY
# MAGIC         a.account_id, a.transaction_id, a.device_id, a.transcation_timestamp_formatted
# MAGIC     )
# MAGIC     select
# MAGIC       transaction_id
# MAGIC     from
# MAGIC       devices_ordered_transaction_history
# MAGIC     where
# MAGIC       total_accounts >= 4
# MAGIC   ),
# MAGIC   remove_flags as(
# MAGIC     select
# MAGIC       transaction_id
# MAGIC     from
# MAGIC       transactions
# MAGIC     where
# MAGIC       has_fake_location = true OR is_emulator = true OR has_root_permissions = true OR app_is_tampered = true
# MAGIC   ),
# MAGIC   remove_big_distances as(
# MAGIC     select
# MAGIC       transaction_id
# MAGIC     from
# MAGIC       transactions
# MAGIC     where
# MAGIC       distance_to_frequent_location > 1000000
# MAGIC   )
# MAGIC  
# MAGIC   select transaction_id
# MAGIC   from has_more_than_4_accounts_per_device
# MAGIC   union
# MAGIC   select transaction_id from remove_flags
# MAGIC   union
# MAGIC   select transaction_id from remove_big_distances
# MAGIC )
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Mid and low risk band table creation

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE medium_low_band AS(
# MAGIC   select
# MAGIC     t.*,
# MAGIC     case when tff.transaction_id is not null then 1 else 0 end as is_fraud,
# MAGIC     case when t.distance_to_frequent_location is null
# MAGIC       OR t.is_emulator is null
# MAGIC       OR t.has_fake_location is null
# MAGIC       OR t.has_root_permissions is null
# MAGIC       OR t.app_is_tampered is null then 1
# MAGIC       else 0
# MAGIC     end as has_null_value
# MAGIC   from
# MAGIC     transactions t
# MAGIC     left anti join high_risk_band hrb on t.transaction_id = hrb.transaction_id
# MAGIC     left join transactions_fraud_feedback tff on t.transaction_id = tff.transaction_id
# MAGIC )
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Putting all bands together

# COMMAND ----------

high_risk_query = '''
select transaction_id from high_risk_band
'''

all_data_query = '''
select t.*, case when tff.transaction_id is not null then 1 else 0 end as is_fraud
from transactions t
left join transactions_fraud_feedback tff on t.transaction_id = tff.transaction_id
'''

df_all = spark.sql(all_data_query).toPandas()
df_high = spark.sql(high_risk_query).toPandas()

df_low_mid = spark.read.table("medium_low_band").select("*").toPandas()

df_low = df_low_mid[(df_low_mid["transaction_value"] < 200) & (df_low_mid["distance_to_frequent_location"] < 10000) & (df_low_mid["device_age_days"] > 0) & (df_low_mid["has_null_value"] == 0)]

df_mid = df_low_mid[~((df_low_mid["transaction_value"] < 200) & (df_low_mid["distance_to_frequent_location"] < 10000) & (df_low_mid["device_age_days"] > 0) & (df_low_mid["has_null_value"] == 0))]


df_low = df_low[["transaction_id"]].copy()
df_low['risk_band'] = "low"

df_mid = df_mid[["transaction_id"]].copy()
df_mid['risk_band'] = "mid"

df_high = df_high[["transaction_id"]]
df_high['risk_band'] = "high"

df_all_bands = pd.concat([df_low, df_mid, df_high])

df_all_with_bands = pd.merge(df_all, df_all_bands, on = "transaction_id", how = 'left')

spark_df = spark.createDataFrame(df_all_with_bands)
spark_df.write.mode("overwrite").saveAsTable("transactions_with_bands")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   risk_band,
# MAGIC   count(transaction_id) as total_transactions
# MAGIC FROM transactions_with_bands
# MAGIC GROUP BY 1
# MAGIC ORDER BY 2

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### How did we get to these rules?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Creating the base

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TABLE transactions_without_high_risk AS(
# MAGIC   with timestamp_formated  as (
# MAGIC     select
# MAGIC       *,
# MAGIC       timestamp_millis(cast(transaction_timestamp as bigint)) as transcation_timestamp_formatted
# MAGIC     from transactions
# MAGIC   ),
# MAGIC
# MAGIC   devices_ordered_transaction_history as (
# MAGIC       SELECT 
# MAGIC           a.account_id, 
# MAGIC           a.transaction_id, 
# MAGIC           a.transcation_timestamp_formatted, 
# MAGIC           a.device_id, 
# MAGIC           count(distinct b.account_id) as CountDistinct
# MAGIC       FROM timestamp_formated a INNER JOIN
# MAGIC       timestamp_formated  b ON a.device_id = b.device_id and 
# MAGIC           b.transcation_timestamp_formatted <= a.transcation_timestamp_formatted
# MAGIC       GROUP BY a.account_id, a.transaction_id, a.device_id, a.transcation_timestamp_formatted
# MAGIC       having CountDistinct > 3
# MAGIC   ),
# MAGIC
# MAGIC   flags_are_true as(
# MAGIC       SELECT
# MAGIC         transaction_id
# MAGIC       from
# MAGIC         transactions
# MAGIC       where
# MAGIC         is_emulator = true OR
# MAGIC         has_fake_location = true or
# MAGIC         has_root_permissions = true or
# MAGIC         app_is_tampered = true
# MAGIC   ),
# MAGIC
# MAGIC   base as (
# MAGIC     select transaction_id from flags_are_true
# MAGIC     union
# MAGIC     select transaction_id from devices_ordered_transaction_history
# MAGIC   ),
# MAGIC
# MAGIC   all_transactions as (
# MAGIC     select
# MAGIC       t.*,
# MAGIC       case when t2.transaction_id is not null then 1 else 0 end as is_fraud
# MAGIC     from
# MAGIC       transactions t
# MAGIC       left join transactions_fraud_feedback t2 on t.transaction_id = t2.transaction_id
# MAGIC   ),
# MAGIC
# MAGIC   all_base as (
# MAGIC     select *
# MAGIC     from
# MAGIC       all_transactions allt
# MAGIC       left anti join base b on allt.transaction_id = b.transaction_id
# MAGIC   )
# MAGIC
# MAGIC
# MAGIC   SELECT * FROM all_base
# MAGIC
# MAGIC )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Distance to Frequent location

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC CASE
# MAGIC   when distance_to_frequent_location between 0 and 1000 then "0 and 1000"
# MAGIC   when distance_to_frequent_location between 1000 and 2000 then "1000 and 2000"
# MAGIC   when distance_to_frequent_location between 2000 and 3000 then "2000 and 3000"
# MAGIC   when distance_to_frequent_location between 3000 and 4000 then "3000 and 4000"
# MAGIC   when distance_to_frequent_location between 4000 and 5000 then "4000 and 5000"
# MAGIC   when distance_to_frequent_location between 5000 and 6000 then "5000 and 6000"
# MAGIC   when distance_to_frequent_location between 6000 and 7000 then "6000 and 7000"
# MAGIC   when distance_to_frequent_location between 7000 and 8000 then "7000 and 8000"
# MAGIC   when distance_to_frequent_location between 8000 and 9000 then "8000 and 9000"
# MAGIC   when distance_to_frequent_location between 9000 and 10000 then "9000 and 10000"
# MAGIC   when distance_to_frequent_location between 10000 and 11000 then "10000 and 11000"
# MAGIC   when distance_to_frequent_location between 11000 and 12000 then "11000 and 12000"
# MAGIC   when distance_to_frequent_location between 12000 and 13000 then "12000 and 13000"
# MAGIC   when distance_to_frequent_location between 13000 and 14000 then "13000 and 14000"
# MAGIC   when distance_to_frequent_location between 14000 and 15000 then "14000 and 15000"
# MAGIC   when distance_to_frequent_location between 15000 and 16000 then "15000 and 16000"
# MAGIC   when distance_to_frequent_location between 16000 and 17000 then "16000 and 17000"
# MAGIC   when distance_to_frequent_location between 17000 and 18000 then "17000 and 18000"
# MAGIC   when distance_to_frequent_location between 18000 and 19000 then "18000 and 19000"
# MAGIC   when distance_to_frequent_location between 19000 and 20000 then "19000 and 20000"
# MAGIC   when distance_to_frequent_location between 20000 and 21000 then "20000 and 21000"
# MAGIC   when distance_to_frequent_location between 21000 and 22000 then "21000 and 22000"
# MAGIC   when distance_to_frequent_location between 22000 and 23000 then "22000 and 23000"
# MAGIC   when distance_to_frequent_location between 23000 and 24000 then "23000 and 24000"
# MAGIC   when distance_to_frequent_location between 24000 and 25000 then "24000 and 25000"
# MAGIC   when distance_to_frequent_location between 25000 and 26000 then "25000 and 26000"
# MAGIC   when distance_to_frequent_location between 26000 and 27000 then "26000 and 27000"
# MAGIC   when distance_to_frequent_location between 27000 and 28000 then "27000 and 28000"
# MAGIC   when distance_to_frequent_location between 28000 and 29000 then "28000 and 29000"
# MAGIC   when distance_to_frequent_location between 29000 and 30000 then "29000 and 30000"
# MAGIC   when distance_to_frequent_location between 30000 and 31000 then "30000 and 31000"
# MAGIC   when distance_to_frequent_location between 31000 and 32000 then "31000 and 32000"
# MAGIC   when distance_to_frequent_location between 32000 and 33000 then "32000 and 33000"
# MAGIC   when distance_to_frequent_location between 33000 and 34000 then "33000 and 34000"
# MAGIC   when distance_to_frequent_location between 34000 and 35000 then "34000 and 35000"
# MAGIC   when distance_to_frequent_location between 35000 and 36000 then "35000 and 36000"
# MAGIC   when distance_to_frequent_location between 36000 and 37000 then "36000 and 37000"
# MAGIC   when distance_to_frequent_location between 37000 and 38000 then "37000 and 38000"
# MAGIC   when distance_to_frequent_location between 38000 and 39000 then "38000 and 39000"
# MAGIC   when distance_to_frequent_location between 39000 and 40000 then "39000 and 40000"
# MAGIC   when distance_to_frequent_location between 40000 and 41000 then "40000 and 41000"
# MAGIC   when distance_to_frequent_location between 41000 and 42000 then "41000 and 42000"
# MAGIC   when distance_to_frequent_location between 42000 and 43000 then "42000 and 43000"
# MAGIC   when distance_to_frequent_location between 43000 and 44000 then "43000 and 44000"
# MAGIC   when distance_to_frequent_location between 44000 and 45000 then "44000 and 45000"
# MAGIC   when distance_to_frequent_location between 45000 and 46000 then "45000 and 46000"
# MAGIC   when distance_to_frequent_location between 46000 and 47000 then "46000 and 47000"
# MAGIC   when distance_to_frequent_location between 47000 and 48000 then "47000 and 48000"
# MAGIC   when distance_to_frequent_location between 48000 and 49000 then "48000 and 49000"
# MAGIC   when distance_to_frequent_location between 49000 and 50000 then "49000 and 50000"
# MAGIC   when distance_to_frequent_location between 50000 and 51000 then "50000 and 51000"
# MAGIC   when distance_to_frequent_location between 51000 and 52000 then "51000 and 52000"
# MAGIC   when distance_to_frequent_location between 52000 and 53000 then "52000 and 53000"
# MAGIC   when distance_to_frequent_location between 53000 and 54000 then "53000 and 54000"
# MAGIC   when distance_to_frequent_location between 54000 and 55000 then "54000 and 55000"
# MAGIC   when distance_to_frequent_location between 55000 and 56000 then "55000 and 56000"
# MAGIC   when distance_to_frequent_location between 56000 and 57000 then "56000 and 57000"
# MAGIC   when distance_to_frequent_location between 57000 and 58000 then "57000 and 58000"
# MAGIC   when distance_to_frequent_location between 58000 and 59000 then "58000 and 59000"
# MAGIC   when distance_to_frequent_location between 59000 and 60000 then "59000 and 60000"
# MAGIC   when distance_to_frequent_location between 60000 and 61000 then "60000 and 61000"
# MAGIC   when distance_to_frequent_location between 61000 and 62000 then "61000 and 62000"
# MAGIC   when distance_to_frequent_location between 62000 and 63000 then "62000 and 63000"
# MAGIC   when distance_to_frequent_location between 63000 and 64000 then "63000 and 64000"
# MAGIC   when distance_to_frequent_location between 64000 and 65000 then "64000 and 65000"
# MAGIC   when distance_to_frequent_location between 65000 and 66000 then "65000 and 66000"
# MAGIC   when distance_to_frequent_location between 66000 and 67000 then "66000 and 67000"
# MAGIC   when distance_to_frequent_location between 67000 and 68000 then "67000 and 68000"
# MAGIC   when distance_to_frequent_location between 68000 and 69000 then "68000 and 69000"
# MAGIC   when distance_to_frequent_location between 69000 and 70000 then "69000 and 70000"
# MAGIC   when distance_to_frequent_location between 70000 and 71000 then "70000 and 71000"
# MAGIC   when distance_to_frequent_location between 71000 and 72000 then "71000 and 72000"
# MAGIC   when distance_to_frequent_location between 72000 and 73000 then "72000 and 73000"
# MAGIC   when distance_to_frequent_location between 73000 and 74000 then "73000 and 74000"
# MAGIC   when distance_to_frequent_location between 74000 and 75000 then "74000 and 75000"
# MAGIC   when distance_to_frequent_location between 75000 and 76000 then "75000 and 76000"
# MAGIC   when distance_to_frequent_location between 76000 and 77000 then "76000 and 77000"
# MAGIC   when distance_to_frequent_location between 77000 and 78000 then "77000 and 78000"
# MAGIC   when distance_to_frequent_location between 78000 and 79000 then "78000 and 79000"
# MAGIC   when distance_to_frequent_location between 79000 and 80000 then "79000 and 80000"
# MAGIC   when distance_to_frequent_location between 80000 and 81000 then "80000 and 81000"
# MAGIC   when distance_to_frequent_location between 81000 and 82000 then "81000 and 82000"
# MAGIC   when distance_to_frequent_location between 82000 and 83000 then "82000 and 83000"
# MAGIC   when distance_to_frequent_location between 83000 and 84000 then "83000 and 84000"
# MAGIC   when distance_to_frequent_location between 84000 and 85000 then "84000 and 85000"
# MAGIC   when distance_to_frequent_location between 85000 and 86000 then "85000 and 86000"
# MAGIC   when distance_to_frequent_location between 86000 and 87000 then "86000 and 87000"
# MAGIC   when distance_to_frequent_location between 87000 and 88000 then "87000 and 88000"
# MAGIC   when distance_to_frequent_location between 88000 and 89000 then "88000 and 89000"
# MAGIC   when distance_to_frequent_location between 89000 and 90000 then "89000 and 90000"
# MAGIC   when distance_to_frequent_location between 90000 and 91000 then "90000 and 91000"
# MAGIC   when distance_to_frequent_location between 91000 and 92000 then "91000 and 92000"
# MAGIC   when distance_to_frequent_location between 92000 and 93000 then "92000 and 93000"
# MAGIC   when distance_to_frequent_location between 93000 and 94000 then "93000 and 94000"
# MAGIC   when distance_to_frequent_location between 94000 and 95000 then "94000 and 95000"
# MAGIC   when distance_to_frequent_location between 95000 and 96000 then "95000 and 96000"
# MAGIC   when distance_to_frequent_location between 96000 and 97000 then "96000 and 97000"
# MAGIC   when distance_to_frequent_location between 97000 and 98000 then "97000 and 98000"
# MAGIC   when distance_to_frequent_location between 98000 and 99000 then "98000 and 99000"
# MAGIC   when distance_to_frequent_location between 99000 and 100000 then  "99000 and 100000"
# MAGIC   when distance_to_frequent_location between 100000 and 200000 then "100000 and 200000"
# MAGIC   when distance_to_frequent_location between 200000 and 300000 then "200000 and 300000"
# MAGIC   when distance_to_frequent_location between 300000 and 400000 then "300000 and 400000"
# MAGIC   when distance_to_frequent_location between 400000 and 500000 then "400000 and 500000"
# MAGIC   when distance_to_frequent_location between 500000 and 600000 then "500000 and 600000"
# MAGIC   when distance_to_frequent_location between 600000 and 700000 then "600000 and 700000"
# MAGIC   when distance_to_frequent_location between 700000 and 800000 then "700000 and 800000"
# MAGIC   when distance_to_frequent_location between 800000 and 900000 then "800000 and 900000"
# MAGIC   when distance_to_frequent_location between 900000 and 1000000 then "900000 and 1000000"
# MAGIC   when distance_to_frequent_location > 1000000 then "> 1000km"
# MAGIC else "unknown" end as distance_break,
# MAGIC   count(distinct transaction_id) as total_transactions,
# MAGIC   SUM(is_fraud) as total_frauds,
# MAGIC   100 * total_frauds / coalesce(total_transactions, 1) as fraud_rate
# MAGIC from transactions_without_high_risk
# MAGIC group by 1
# MAGIC order by
# MAGIC   case when distance_break = "0 and 1000" then 1
# MAGIC   when distance_break = "1000 and 2000" then 2
# MAGIC   when distance_break = "2000 and 3000" then 3
# MAGIC   when distance_break = "3000 and 4000" then 4
# MAGIC   when distance_break = "4000 and 5000" then 5
# MAGIC   when distance_break = "5000 and 6000" then 6
# MAGIC   when distance_break = "6000 and 7000" then 7
# MAGIC   when distance_break = "7000 and 8000" then 8
# MAGIC   when distance_break = "8000 and 9000" then 9
# MAGIC   when distance_break = "9000 and 10000" then 10
# MAGIC   when distance_break = "10000 and 11000" then 11
# MAGIC   when distance_break = "11000 and 12000" then 12
# MAGIC   when distance_break = "12000 and 13000" then 13
# MAGIC   when distance_break = "13000 and 14000" then 14
# MAGIC   when distance_break = "14000 and 15000" then 15
# MAGIC   when distance_break = "15000 and 16000" then 16
# MAGIC   when distance_break = "16000 and 17000" then 17
# MAGIC   when distance_break = "17000 and 18000" then 18
# MAGIC   when distance_break = "18000 and 19000" then 19
# MAGIC   when distance_break = "19000 and 20000" then 20
# MAGIC   when distance_break = "20000 and 21000" then 21
# MAGIC   when distance_break = "21000 and 22000" then 22
# MAGIC   when distance_break = "22000 and 23000" then 23
# MAGIC   when distance_break = "23000 and 24000" then 24
# MAGIC   when distance_break = "24000 and 25000" then 25
# MAGIC   when distance_break = "25000 and 26000" then 26
# MAGIC   when distance_break = "26000 and 27000" then 27
# MAGIC   when distance_break = "27000 and 28000" then 28
# MAGIC   when distance_break = "28000 and 29000" then 29
# MAGIC   when distance_break = "29000 and 30000" then 30
# MAGIC   when distance_break = "30000 and 31000" then 31
# MAGIC   when distance_break = "31000 and 32000" then 32
# MAGIC   when distance_break = "32000 and 33000" then 33
# MAGIC   when distance_break = "33000 and 34000" then 34
# MAGIC   when distance_break = "34000 and 35000" then 35
# MAGIC   when distance_break = "35000 and 36000" then 36
# MAGIC   when distance_break = "36000 and 37000" then 37
# MAGIC   when distance_break = "37000 and 38000" then 38
# MAGIC   when distance_break = "38000 and 39000" then 39
# MAGIC   when distance_break = "39000 and 40000" then 40
# MAGIC   when distance_break = "40000 and 41000" then 41
# MAGIC   when distance_break = "41000 and 42000" then 42
# MAGIC   when distance_break = "42000 and 43000" then 43
# MAGIC   when distance_break = "43000 and 44000" then 44
# MAGIC   when distance_break = "44000 and 45000" then 45
# MAGIC   when distance_break = "45000 and 46000" then 46
# MAGIC   when distance_break = "46000 and 47000" then 47
# MAGIC   when distance_break = "47000 and 48000" then 48
# MAGIC   when distance_break = "48000 and 49000" then 49
# MAGIC   when distance_break = "49000 and 50000" then 50
# MAGIC   when distance_break = "50000 and 51000" then 51
# MAGIC   when distance_break = "51000 and 52000" then 52
# MAGIC   when distance_break = "52000 and 53000" then 53
# MAGIC   when distance_break = "53000 and 54000" then 54
# MAGIC   when distance_break = "54000 and 55000" then 55
# MAGIC   when distance_break = "55000 and 56000" then 56
# MAGIC   when distance_break = "56000 and 57000" then 57
# MAGIC   when distance_break = "57000 and 58000" then 58
# MAGIC   when distance_break = "58000 and 59000" then 59
# MAGIC   when distance_break = "59000 and 60000" then 60
# MAGIC   when distance_break = "60000 and 61000" then 61
# MAGIC   when distance_break = "61000 and 62000" then 62
# MAGIC   when distance_break = "62000 and 63000" then 63
# MAGIC   when distance_break = "63000 and 64000" then 64
# MAGIC   when distance_break = "64000 and 65000" then 65
# MAGIC   when distance_break = "65000 and 66000" then 66
# MAGIC   when distance_break = "66000 and 67000" then 67
# MAGIC   when distance_break = "67000 and 68000" then 68
# MAGIC   when distance_break = "68000 and 69000" then 69
# MAGIC   when distance_break = "69000 and 70000" then 70
# MAGIC   when distance_break = "70000 and 71000" then 71
# MAGIC   when distance_break = "71000 and 72000" then 72
# MAGIC   when distance_break = "72000 and 73000" then 73
# MAGIC   when distance_break = "73000 and 74000" then 74
# MAGIC   when distance_break = "74000 and 75000" then 75
# MAGIC   when distance_break = "75000 and 76000" then 76
# MAGIC   when distance_break = "76000 and 77000" then 77
# MAGIC   when distance_break = "77000 and 78000" then 78
# MAGIC   when distance_break = "78000 and 79000" then 79
# MAGIC   when distance_break = "79000 and 80000" then 80
# MAGIC   when distance_break = "80000 and 81000" then 81
# MAGIC   when distance_break = "81000 and 82000" then 82
# MAGIC   when distance_break = "82000 and 83000" then 83
# MAGIC   when distance_break = "83000 and 84000" then 84
# MAGIC   when distance_break = "84000 and 85000" then 85
# MAGIC   when distance_break = "85000 and 86000" then 86
# MAGIC   when distance_break = "86000 and 87000" then 87
# MAGIC   when distance_break = "87000 and 88000" then 88
# MAGIC   when distance_break = "88000 and 89000" then 89
# MAGIC   when distance_break = "89000 and 90000" then 90
# MAGIC   when distance_break = "90000 and 91000" then 91
# MAGIC   when distance_break = "91000 and 92000" then 92
# MAGIC   when distance_break = "92000 and 93000" then 93
# MAGIC   when distance_break = "93000 and 94000" then 94
# MAGIC   when distance_break = "94000 and 95000" then 95
# MAGIC   when distance_break = "95000 and 96000" then 96
# MAGIC   when distance_break = "96000 and 97000" then 97
# MAGIC   when distance_break = "97000 and 98000" then 98
# MAGIC   when distance_break = "98000 and 99000" then 99
# MAGIC   when distance_break = "99000 and 100000" then 100
# MAGIC   when distance_break = "100000 and 200000" then 101
# MAGIC   when distance_break = "200000 and 300000" then 102
# MAGIC   when distance_break = "300000 and 400000" then 103
# MAGIC   when distance_break = "400000 and 500000" then 104
# MAGIC   when distance_break = "500000 and 600000" then 105
# MAGIC   when distance_break = "600000 and 700000" then 106
# MAGIC   when distance_break = "700000 and 800000" then 107
# MAGIC   when distance_break = "800000 and 900000" then 108
# MAGIC   when distance_break = "900000 and 1000000" then 109
# MAGIC   else 110
# MAGIC end

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Transaction Value

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC   CASE
# MAGIC     when transaction_value between 0 and 10 then "0 and 10"
# MAGIC     when transaction_value between 10 and 20 then "10 and 20"
# MAGIC     when transaction_value between 20 and 30 then "20 and 30"
# MAGIC     when transaction_value between 30 and 40 then "30 and 40"
# MAGIC     when transaction_value between 40 and 50 then "40 and 50"
# MAGIC     when transaction_value between 50 and 60 then "50 and 60"
# MAGIC     when transaction_value between 60 and 70 then "60 and 70"
# MAGIC     when transaction_value between 70 and 80 then "70 and 80"
# MAGIC     when transaction_value between 80 and 90 then "80 and 90"
# MAGIC     when transaction_value between 90 and 100 then "90 and 100"
# MAGIC     when transaction_value between 100 and 110 then "100 and 110"
# MAGIC     when transaction_value between 110 and 120 then "110 and 120"
# MAGIC     when transaction_value between 120 and 130 then "120 and 130"
# MAGIC     when transaction_value between 130 and 140 then "130 and 140"
# MAGIC     when transaction_value between 140 and 150 then "140 and 150"
# MAGIC     when transaction_value between 150 and 160 then "150 and 160"
# MAGIC     when transaction_value between 160 and 170 then "160 and 170"
# MAGIC     when transaction_value between 170 and 180 then "170 and 180"
# MAGIC     when transaction_value between 180 and 190 then "180 and 190"
# MAGIC     when transaction_value between 190 and 200 then "190 and 200"
# MAGIC     when transaction_value between 200 and 210 then "200 and 210"
# MAGIC     when transaction_value between 210 and 220 then "210 and 220"
# MAGIC     when transaction_value between 220 and 230 then "220 and 230"
# MAGIC     when transaction_value between 230 and 240 then "230 and 240"
# MAGIC     when transaction_value between 240 and 250 then "240 and 250"
# MAGIC     when transaction_value between 250 and 260 then "250 and 260"
# MAGIC     when transaction_value between 260 and 270 then "260 and 270"
# MAGIC     when transaction_value between 270 and 280 then "270 and 280"
# MAGIC     when transaction_value between 280 and 290 then "280 and 290"
# MAGIC     when transaction_value between 290 and 300 then "290 and 300"
# MAGIC     when transaction_value between 300 and 310 then "300 and 310"
# MAGIC     when transaction_value between 310 and 320 then "310 and 320"
# MAGIC     when transaction_value between 320 and 330 then "320 and 330"
# MAGIC     when transaction_value between 330 and 340 then "330 and 340"
# MAGIC     when transaction_value between 340 and 350 then "340 and 350"
# MAGIC     when transaction_value between 350 and 360 then "350 and 360"
# MAGIC     when transaction_value between 360 and 370 then "360 and 370"
# MAGIC     when transaction_value between 370 and 380 then "370 and 380"
# MAGIC     when transaction_value between 380 and 390 then "380 and 390"
# MAGIC     when transaction_value between 390 and 400 then "390 and 400"
# MAGIC     when transaction_value between 400 and 410 then "400 and 410"
# MAGIC     when transaction_value between 410 and 420 then "410 and 420"
# MAGIC     when transaction_value between 420 and 430 then "420 and 430"
# MAGIC     when transaction_value between 430 and 440 then "430 and 440"
# MAGIC     when transaction_value between 440 and 450 then "440 and 450"
# MAGIC     when transaction_value between 450 and 460 then "450 and 460"
# MAGIC     when transaction_value between 460 and 470 then "460 and 470"
# MAGIC     when transaction_value between 470 and 480 then "470 and 480"
# MAGIC     when transaction_value between 480 and 490 then "480 and 490"
# MAGIC     when transaction_value between 490 and 500 then "490 and 500"
# MAGIC     when transaction_value between 500 and 510 then "500 and 510"
# MAGIC     when transaction_value between 510 and 520 then "510 and 520"
# MAGIC     when transaction_value between 520 and 530 then "520 and 530"
# MAGIC     when transaction_value between 530 and 540 then "530 and 540"
# MAGIC     when transaction_value between 540 and 550 then "540 and 550"
# MAGIC     when transaction_value between 550 and 560 then "550 and 560"
# MAGIC     when transaction_value between 560 and 570 then "560 and 570"
# MAGIC     when transaction_value between 570 and 580 then "570 and 580"
# MAGIC     when transaction_value between 580 and 590 then "580 and 590"
# MAGIC     when transaction_value between 590 and 600 then "590 and 600"
# MAGIC     when transaction_value between 600 and 610 then "600 and 610"
# MAGIC     when transaction_value between 610 and 620 then "610 and 620"
# MAGIC     when transaction_value between 620 and 630 then "620 and 630"
# MAGIC     when transaction_value between 630 and 640 then "630 and 640"
# MAGIC     when transaction_value between 640 and 650 then "640 and 650"
# MAGIC     when transaction_value between 650 and 660 then "650 and 660"
# MAGIC     when transaction_value between 660 and 670 then "660 and 670"
# MAGIC     when transaction_value between 670 and 680 then "670 and 680"
# MAGIC     when transaction_value between 680 and 690 then "680 and 690"
# MAGIC     when transaction_value between 690 and 700 then "690 and 700"
# MAGIC     when transaction_value between 700 and 710 then "700 and 710"
# MAGIC     when transaction_value between 710 and 720 then "710 and 720"
# MAGIC     when transaction_value between 720 and 730 then "720 and 730"
# MAGIC     when transaction_value between 730 and 740 then "730 and 740"
# MAGIC     when transaction_value between 740 and 750 then "740 and 750"
# MAGIC     when transaction_value between 750 and 760 then "750 and 760"
# MAGIC     when transaction_value between 760 and 770 then "760 and 770"
# MAGIC     when transaction_value between 770 and 780 then "770 and 780"
# MAGIC     when transaction_value between 780 and 790 then "780 and 790"
# MAGIC     when transaction_value between 790 and 800 then "790 and 800"
# MAGIC     when transaction_value between 800 and 810 then "800 and 810"
# MAGIC     when transaction_value between 810 and 820 then "810 and 820"
# MAGIC     when transaction_value between 820 and 830 then "820 and 830"
# MAGIC     when transaction_value between 830 and 840 then "830 and 840"
# MAGIC     when transaction_value between 840 and 850 then "840 and 850"
# MAGIC     when transaction_value between 850 and 860 then "850 and 860"
# MAGIC     when transaction_value between 860 and 870 then "860 and 870"
# MAGIC     when transaction_value between 870 and 880 then "870 and 880"
# MAGIC     when transaction_value between 880 and 890 then "880 and 890"
# MAGIC     when transaction_value between 890 and 900 then "890 and 900"
# MAGIC     when transaction_value between 900 and 910 then "900 and 910"
# MAGIC     when transaction_value between 910 and 920 then "910 and 920"
# MAGIC     when transaction_value between 920 and 930 then "920 and 930"
# MAGIC     when transaction_value between 930 and 940 then "930 and 940"
# MAGIC     when transaction_value between 940 and 950 then "940 and 950"
# MAGIC     when transaction_value between 950 and 960 then "950 and 960"
# MAGIC     when transaction_value between 960 and 970 then "960 and 970"
# MAGIC     when transaction_value between 970 and 980 then "970 and 980"
# MAGIC     when transaction_value between 980 and 990 then "980 and 990"
# MAGIC     when transaction_value between 990 and 1000 then  "990 and 1000"
# MAGIC     when transaction_value between 1000 and 2000 then "1000 and 2000"
# MAGIC     when transaction_value between 2000 and 3000 then "2000 and 3000"
# MAGIC     when transaction_value between 3000 and 4000 then "3000 and 4000"
# MAGIC     when transaction_value between 4000 and 5000 then "4000 and 5000"
# MAGIC     when transaction_value between 5000 and 6000 then "5000 and 6000"
# MAGIC     when transaction_value between 6000 and 7000 then "6000 and 7000"
# MAGIC     when transaction_value between 7000 and 8000 then "7000 and 8000"
# MAGIC     when transaction_value between 8000 and 9000 then "8000 and 9000"
# MAGIC     when transaction_value between 9000 and 10000 then "9000 and 10000"
# MAGIC     when transaction_value > 10000 then "> 10000"
# MAGIC   else "unknown" end as value_break,
# MAGIC   count(distinct transaction_id) as total_transactions,
# MAGIC   SUM(is_fraud) as total_frauds,
# MAGIC   100 * total_frauds / coalesce(total_transactions, 1) as fraud_rate
# MAGIC from transactions_without_high_risk
# MAGIC where distance_to_frequent_location < 100000
# MAGIC group by 1
# MAGIC order by
# MAGIC   case when value_break = "0 and 10" then 1
# MAGIC   when value_break = "10 and 20" then 2
# MAGIC   when value_break = "20 and 30" then 3
# MAGIC   when value_break = "30 and 40" then 4
# MAGIC   when value_break = "40 and 50" then 5
# MAGIC   when value_break = "50 and 60" then 6
# MAGIC   when value_break = "60 and 70" then 7
# MAGIC   when value_break = "70 and 80" then 8
# MAGIC   when value_break = "80 and 90" then 9
# MAGIC   when value_break = "90 and 100" then 10
# MAGIC   when value_break = "100 and 110" then 11
# MAGIC   when value_break = "110 and 120" then 12
# MAGIC   when value_break = "120 and 130" then 13
# MAGIC   when value_break = "130 and 140" then 14
# MAGIC   when value_break = "140 and 150" then 15
# MAGIC   when value_break = "150 and 160" then 16
# MAGIC   when value_break = "160 and 170" then 17
# MAGIC   when value_break = "170 and 180" then 18
# MAGIC   when value_break = "180 and 190" then 19
# MAGIC   when value_break = "190 and 200" then 20
# MAGIC   when value_break = "200 and 210" then 21
# MAGIC   when value_break = "210 and 220" then 22
# MAGIC   when value_break = "220 and 230" then 23
# MAGIC   when value_break = "230 and 240" then 24
# MAGIC   when value_break = "240 and 250" then 25
# MAGIC   when value_break = "250 and 260" then 26
# MAGIC   when value_break = "260 and 270" then 27
# MAGIC   when value_break = "270 and 280" then 28
# MAGIC   when value_break = "280 and 290" then 29
# MAGIC   when value_break = "290 and 300" then 30
# MAGIC   when value_break = "300 and 310" then 31
# MAGIC   when value_break = "310 and 320" then 32
# MAGIC   when value_break = "320 and 330" then 33
# MAGIC   when value_break = "330 and 340" then 34
# MAGIC   when value_break = "340 and 350" then 35
# MAGIC   when value_break = "350 and 360" then 36
# MAGIC   when value_break = "360 and 370" then 37
# MAGIC   when value_break = "370 and 380" then 38
# MAGIC   when value_break = "380 and 390" then 39
# MAGIC   when value_break = "390 and 400" then 40
# MAGIC   when value_break = "400 and 410" then 41
# MAGIC   when value_break = "410 and 420" then 42
# MAGIC   when value_break = "420 and 430" then 43
# MAGIC   when value_break = "430 and 440" then 44
# MAGIC   when value_break = "440 and 450" then 45
# MAGIC   when value_break = "450 and 460" then 46
# MAGIC   when value_break = "460 and 470" then 47
# MAGIC   when value_break = "470 and 480" then 48
# MAGIC   when value_break = "480 and 490" then 49
# MAGIC   when value_break = "490 and 500" then 50
# MAGIC   when value_break = "500 and 510" then 51
# MAGIC   when value_break = "510 and 520" then 52
# MAGIC   when value_break = "520 and 530" then 53
# MAGIC   when value_break = "530 and 540" then 54
# MAGIC   when value_break = "540 and 550" then 55
# MAGIC   when value_break = "550 and 560" then 56
# MAGIC   when value_break = "560 and 570" then 57
# MAGIC   when value_break = "570 and 580" then 58
# MAGIC   when value_break = "580 and 590" then 59
# MAGIC   when value_break = "590 and 600" then 60
# MAGIC   when value_break = "600 and 610" then 61
# MAGIC   when value_break = "610 and 620" then 62
# MAGIC   when value_break = "620 and 630" then 63
# MAGIC   when value_break = "630 and 640" then 64
# MAGIC   when value_break = "640 and 650" then 65
# MAGIC   when value_break = "650 and 660" then 66
# MAGIC   when value_break = "660 and 670" then 67
# MAGIC   when value_break = "670 and 680" then 68
# MAGIC   when value_break = "680 and 690" then 69
# MAGIC   when value_break = "690 and 700" then 70
# MAGIC   when value_break = "700 and 710" then 71
# MAGIC   when value_break = "710 and 720" then 72
# MAGIC   when value_break = "720 and 730" then 73
# MAGIC   when value_break = "730 and 740" then 74
# MAGIC   when value_break = "740 and 750" then 75
# MAGIC   when value_break = "750 and 760" then 76
# MAGIC   when value_break = "760 and 770" then 77
# MAGIC   when value_break = "770 and 780" then 78
# MAGIC   when value_break = "780 and 790" then 79
# MAGIC   when value_break = "790 and 800" then 80
# MAGIC   when value_break = "800 and 810" then 81
# MAGIC   when value_break = "810 and 820" then 82
# MAGIC   when value_break = "820 and 830" then 83
# MAGIC   when value_break = "830 and 840" then 84
# MAGIC   when value_break = "840 and 850" then 85
# MAGIC   when value_break = "850 and 860" then 86
# MAGIC   when value_break = "860 and 870" then 87
# MAGIC   when value_break = "870 and 880" then 88
# MAGIC   when value_break = "880 and 890" then 89
# MAGIC   when value_break = "890 and 900" then 90
# MAGIC   when value_break = "900 and 910" then 91
# MAGIC   when value_break = "910 and 920" then 92
# MAGIC   when value_break = "920 and 930" then 93
# MAGIC   when value_break = "930 and 940" then 94
# MAGIC   when value_break = "940 and 950" then 95
# MAGIC   when value_break = "950 and 960" then 96
# MAGIC   when value_break = "960 and 970" then 97
# MAGIC   when value_break = "970 and 980" then 98
# MAGIC   when value_break = "980 and 990" then 99
# MAGIC   when value_break = "990 and 1000" then 100
# MAGIC   when value_break = "1000 and 2000" then 101
# MAGIC   when value_break = "2000 and 3000" then 102
# MAGIC   when value_break = "3000 and 4000" then 103
# MAGIC   when value_break = "4000 and 5000" then 104
# MAGIC   when value_break = "5000 and 6000" then 105
# MAGIC   when value_break = "6000 and 7000" then 106
# MAGIC   when value_break = "7000 and 8000" then 107
# MAGIC   when value_break = "8000 and 9000" then 108
# MAGIC   when value_break = "9000 and 10000" then 109
# MAGIC   else 110
# MAGIC end

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Device Age Days

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC SELECT
# MAGIC CASE
# MAGIC   when device_age_days between 0 and 1 then "0 and 1"
# MAGIC   when device_age_days between 1 and 2 then "1 and 2"
# MAGIC   when device_age_days between 2 and 3 then "2 and 3"
# MAGIC   when device_age_days between 3 and 4 then "3 and 4"
# MAGIC   when device_age_days between 4 and 5 then "4 and 5"
# MAGIC   when device_age_days between 5 and 6 then "5 and 6"
# MAGIC   when device_age_days between 6 and 7 then "6 and 7"
# MAGIC   when device_age_days between 7 and 8 then "7 and 8"
# MAGIC   when device_age_days between 8 and 9 then "8 and 9"
# MAGIC   when device_age_days between 9 and 10 then "9 and 10"
# MAGIC   when device_age_days between 11 and 16 then "11 and 16"
# MAGIC   when device_age_days between 16 and 21 then "16 and 21"
# MAGIC   when device_age_days between 21 and 26 then "21 and 26"
# MAGIC   when device_age_days between 26 and 31 then "26 and 31"
# MAGIC   when device_age_days between 31 and 36 then "31 and 36"
# MAGIC   when device_age_days between 36 and 41 then "36 and 41"
# MAGIC   when device_age_days between 41 and 46 then "41 and 46"
# MAGIC   when device_age_days between 46 and 51 then "46 and 51"
# MAGIC   when device_age_days between 51 and 56 then "51 and 56"
# MAGIC   when device_age_days between 56 and 61 then "56 and 61"
# MAGIC   when device_age_days between 61 and 66 then "61 and 66"
# MAGIC   when device_age_days between 66 and 71 then "66 and 71"
# MAGIC   when device_age_days between 71 and 76 then "71 and 76"
# MAGIC   when device_age_days between 76 and 81 then "76 and 81"
# MAGIC   when device_age_days between 81 and 86 then "81 and 86"
# MAGIC   when device_age_days between 86 and 91 then "86 and 91"
# MAGIC   when device_age_days between 91 and 96 then "91 and 96"
# MAGIC   when device_age_days between 96 and 101 then "96 and 101"
# MAGIC   when device_age_days between 101 and 106 then "101 and 106"
# MAGIC   when device_age_days between 106 and 111 then "106 and 111"
# MAGIC   when device_age_days between 111 and 116 then "111 and 116"
# MAGIC   when device_age_days between 116 and 121 then "116 and 121"
# MAGIC   when device_age_days between 121 and 126 then "121 and 126"
# MAGIC   when device_age_days between 126 and 131 then "126 and 131"
# MAGIC   when device_age_days between 131 and 136 then "131 and 136"
# MAGIC   when device_age_days between 136 and 141 then "136 and 141"
# MAGIC   when device_age_days between 141 and 146 then "141 and 146"
# MAGIC   when device_age_days between 146 and 151 then "146 and 151"
# MAGIC   when device_age_days between 151 and 156 then "151 and 156"
# MAGIC   when device_age_days between 156 and 161 then "156 and 161"
# MAGIC   when device_age_days between 161 and 166 then "161 and 166"
# MAGIC   when device_age_days between 166 and 171 then "166 and 171"
# MAGIC   when device_age_days between 171 and 176 then "171 and 176"
# MAGIC   when device_age_days between 176 and 181 then "176 and 181"
# MAGIC   when device_age_days between 181 and 186 then "181 and 186"
# MAGIC   when device_age_days between 186 and 191 then "186 and 191"
# MAGIC   when device_age_days between 191 and 196 then "191 and 196"
# MAGIC   when device_age_days between 196 and 201 then "196 and 201"
# MAGIC   when device_age_days between 201 and 206 then "201 and 206"
# MAGIC   when device_age_days between 206 and 211 then "206 and 211"
# MAGIC   when device_age_days between 211 and 216 then "211 and 216"
# MAGIC   when device_age_days between 216 and 221 then "216 and 221"
# MAGIC   when device_age_days between 221 and 226 then "221 and 226"
# MAGIC   when device_age_days between 226 and 231 then "226 and 231"
# MAGIC   when device_age_days between 231 and 236 then "231 and 236"
# MAGIC   when device_age_days between 236 and 241 then "236 and 241"
# MAGIC   when device_age_days between 241 and 246 then "241 and 246"
# MAGIC   when device_age_days between 246 and 251 then "246 and 251"
# MAGIC   when device_age_days between 251 and 256 then "251 and 256"
# MAGIC   when device_age_days between 256 and 261 then "256 and 261"
# MAGIC   when device_age_days between 261 and 266 then "261 and 266"
# MAGIC   when device_age_days between 266 and 271 then "266 and 271"
# MAGIC   when device_age_days between 271 and 276 then "271 and 276"
# MAGIC   when device_age_days between 276 and 281 then "276 and 281"
# MAGIC   when device_age_days between 281 and 286 then "281 and 286"
# MAGIC   when device_age_days between 286 and 291 then "286 and 291"
# MAGIC   when device_age_days between 291 and 296 then "291 and 296"
# MAGIC   when device_age_days between 296 and 301 then "296 and 301"
# MAGIC   when device_age_days between 301 and 306 then "301 and 306"
# MAGIC   when device_age_days between 306 and 311 then "306 and 311"
# MAGIC   when device_age_days between 311 and 316 then "311 and 316"
# MAGIC   when device_age_days between 316 and 321 then "316 and 321"
# MAGIC   when device_age_days between 321 and 326 then "321 and 326"
# MAGIC   when device_age_days between 326 and 331 then "326 and 331"
# MAGIC   when device_age_days between 331 and 336 then "331 and 336"
# MAGIC   when device_age_days between 336 and 341 then "336 and 341"
# MAGIC   when device_age_days between 341 and 346 then "341 and 346"
# MAGIC   when device_age_days between 346 and 351 then "346 and 351"
# MAGIC   when device_age_days between 351 and 356 then "351 and 356"
# MAGIC   when device_age_days between 356 and 361 then "356 and 361"
# MAGIC   when device_age_days between 361 and 366 then "361 and 366"
# MAGIC   when device_age_days between 366 and 371 then "366 and 371"
# MAGIC   when device_age_days between 371 and 376 then "371 and 376"
# MAGIC   when device_age_days between 376 and 381 then "376 and 381"
# MAGIC   when device_age_days between 381 and 386 then "381 and 386"
# MAGIC   when device_age_days between 386 and 391 then "386 and 391"
# MAGIC   when device_age_days between 391 and 396 then "391 and 396"
# MAGIC   when device_age_days between 396 and 401 then "396 and 401"
# MAGIC   when device_age_days between 401 and 406 then "401 and 406"
# MAGIC   when device_age_days between 406 and 411 then "406 and 411"
# MAGIC   when device_age_days between 411 and 416 then "411 and 416"
# MAGIC   when device_age_days between 416 and 421 then "416 and 421"
# MAGIC   when device_age_days between 421 and 426 then "421 and 426"
# MAGIC   when device_age_days between 426 and 431 then "426 and 431"
# MAGIC   when device_age_days between 431 and 436 then "431 and 436"
# MAGIC   when device_age_days between 436 and 441 then "436 and 441"
# MAGIC   when device_age_days between 441 and 446 then "441 and 446"
# MAGIC   when device_age_days between 446 and 451 then "446 and 451"
# MAGIC   when device_age_days between 451 and 456 then "451 and 456"
# MAGIC   when device_age_days between 456 and 461 then "456 and 461"
# MAGIC   when device_age_days between 461 and 466 then "461 and 466"
# MAGIC   when device_age_days between 466 and 471 then "466 and 471"
# MAGIC   when device_age_days between 471 and 476 then "471 and 476"
# MAGIC   when device_age_days between 476 and 481 then "476 and 481"
# MAGIC   when device_age_days between 481 and 486 then "481 and 486"
# MAGIC   when device_age_days between 486 and 491 then "486 and 491"
# MAGIC   when device_age_days between 491 and 496 then "491 and 496"
# MAGIC   when device_age_days between 496 and 500 then "496 and 500"
# MAGIC   when device_age_days between 501 and 506 then "501 and 506"
# MAGIC   when device_age_days between 506 and 511 then "506 and 511"
# MAGIC   when device_age_days between 511 and 516 then "511 and 516"
# MAGIC   when device_age_days between 516 and 521 then "516 and 521"
# MAGIC   when device_age_days between 521 and 526 then "521 and 526"
# MAGIC   when device_age_days between 526 and 531 then "526 and 531"
# MAGIC   when device_age_days between 531 and 536 then "531 and 536"
# MAGIC   when device_age_days between 536 and 541 then "536 and 541"
# MAGIC   when device_age_days between 541 and 546 then "541 and 546"
# MAGIC   when device_age_days between 546 and 551 then "546 and 551"
# MAGIC   when device_age_days between 551 and 556 then "551 and 556"
# MAGIC   when device_age_days between 556 and 561 then "556 and 561"
# MAGIC   when device_age_days between 561 and 566 then "561 and 566"
# MAGIC   when device_age_days between 566 and 571 then "566 and 571"
# MAGIC   when device_age_days between 571 and 576 then "571 and 576"
# MAGIC   when device_age_days between 576 and 581 then "576 and 581"
# MAGIC   when device_age_days > 581 then "> 581"
# MAGIC else "unknown" end as age_break,
# MAGIC   count(distinct transaction_id) as total_transactions,
# MAGIC   SUM(is_fraud) as total_frauds,
# MAGIC   100 * total_frauds / coalesce(total_transactions, 1) as fraud_rate
# MAGIC from transactions_without_high_risk
# MAGIC group by 1
# MAGIC order by
# MAGIC   case
# MAGIC   when age_break = "0 and 1" then 1
# MAGIC   when age_break = "1 and 2" then 2
# MAGIC   when age_break = "2 and 3" then 3
# MAGIC   when age_break = "3 and 4" then 4
# MAGIC   when age_break = "4 and 5" then 5
# MAGIC   when age_break = "5 and 6" then 6
# MAGIC   when age_break = "6 and 7" then 7
# MAGIC   when age_break = "7 and 8" then 8
# MAGIC   when age_break = "8 and 9" then 9
# MAGIC   when age_break = "9 and 10" then 10
# MAGIC   when age_break = "11 and 16" then 11
# MAGIC   when age_break = "16 and 21" then 12
# MAGIC   when age_break = "21 and 26" then 13
# MAGIC   when age_break = "26 and 31" then 14
# MAGIC   when age_break = "31 and 36" then 15
# MAGIC   when age_break = "36 and 41" then 16
# MAGIC   when age_break = "41 and 46" then 17
# MAGIC   when age_break = "46 and 51" then 18
# MAGIC   when age_break = "51 and 56" then 19
# MAGIC   when age_break = "56 and 61" then 20
# MAGIC   when age_break = "61 and 66" then 21
# MAGIC   when age_break = "66 and 71" then 22
# MAGIC   when age_break = "71 and 76" then 23
# MAGIC   when age_break = "76 and 81" then 24
# MAGIC   when age_break = "81 and 86" then 25
# MAGIC   when age_break = "86 and 91" then 26
# MAGIC   when age_break = "91 and 96" then 27
# MAGIC   when age_break = "96 and 101" then 28
# MAGIC   when age_break = "101 and 106" then 29
# MAGIC   when age_break = "106 and 111" then 30
# MAGIC   when age_break = "111 and 116" then 31
# MAGIC   when age_break = "116 and 121" then 32
# MAGIC   when age_break = "121 and 126" then 33
# MAGIC   when age_break = "126 and 131" then 34
# MAGIC   when age_break = "131 and 136" then 35
# MAGIC   when age_break = "136 and 141" then 36
# MAGIC   when age_break = "141 and 146" then 37
# MAGIC   when age_break = "146 and 151" then 38
# MAGIC   when age_break = "151 and 156" then 39
# MAGIC   when age_break = "156 and 161" then 40
# MAGIC   when age_break = "161 and 166" then 41
# MAGIC   when age_break = "166 and 171" then 42
# MAGIC   when age_break = "171 and 176" then 43
# MAGIC   when age_break = "176 and 181" then 44
# MAGIC   when age_break = "181 and 186" then 45
# MAGIC   when age_break = "186 and 191" then 46
# MAGIC   when age_break = "191 and 196" then 47
# MAGIC   when age_break = "196 and 201" then 48
# MAGIC   when age_break = "201 and 206" then 49
# MAGIC   when age_break = "206 and 211" then 50
# MAGIC   when age_break = "211 and 216" then 51
# MAGIC   when age_break = "216 and 221" then 52
# MAGIC   when age_break = "221 and 226" then 53
# MAGIC   when age_break = "226 and 231" then 54
# MAGIC   when age_break = "231 and 236" then 55
# MAGIC   when age_break = "236 and 241" then 56
# MAGIC   when age_break = "241 and 246" then 57
# MAGIC   when age_break = "246 and 251" then 58
# MAGIC   when age_break = "251 and 256" then 59
# MAGIC   when age_break = "256 and 261" then 60
# MAGIC   when age_break = "261 and 266" then 61
# MAGIC   when age_break = "266 and 271" then 62
# MAGIC   when age_break = "271 and 276" then 63
# MAGIC   when age_break = "276 and 281" then 64
# MAGIC   when age_break = "281 and 286" then 65
# MAGIC   when age_break = "286 and 291" then 66
# MAGIC   when age_break = "291 and 296" then 67
# MAGIC   when age_break = "296 and 301" then 68
# MAGIC   when age_break = "301 and 306" then 69
# MAGIC   when age_break = "306 and 311" then 70
# MAGIC   when age_break = "311 and 316" then 71
# MAGIC   when age_break = "316 and 321" then 72
# MAGIC   when age_break = "321 and 326" then 73
# MAGIC   when age_break = "326 and 331" then 74
# MAGIC   when age_break = "331 and 336" then 75
# MAGIC   when age_break = "336 and 341" then 76
# MAGIC   when age_break = "341 and 346" then 77
# MAGIC   when age_break = "346 and 351" then 78
# MAGIC   when age_break = "351 and 356" then 79
# MAGIC   when age_break = "356 and 361" then 80
# MAGIC   when age_break = "361 and 366" then 81
# MAGIC   when age_break = "366 and 371" then 82
# MAGIC   when age_break = "371 and 376" then 83
# MAGIC   when age_break = "376 and 381" then 84
# MAGIC   when age_break = "381 and 386" then 85
# MAGIC   when age_break = "386 and 391" then 86
# MAGIC   when age_break = "391 and 396" then 87
# MAGIC   when age_break = "396 and 401" then 88
# MAGIC   when age_break = "401 and 406" then 89
# MAGIC   when age_break = "406 and 411" then 90
# MAGIC   when age_break = "411 and 416" then 91
# MAGIC   when age_break = "416 and 421" then 92
# MAGIC   when age_break = "421 and 426" then 93
# MAGIC   when age_break = "426 and 431" then 94
# MAGIC   when age_break = "431 and 436" then 95
# MAGIC   when age_break = "436 and 441" then 96
# MAGIC   when age_break = "441 and 446" then 97
# MAGIC   when age_break = "446 and 451" then 98
# MAGIC   when age_break = "451 and 456" then 99
# MAGIC   when age_break = "456 and 461" then 100
# MAGIC   when age_break = "461 and 466" then 101
# MAGIC   when age_break = "466 and 471" then 102
# MAGIC   when age_break = "471 and 476" then 103
# MAGIC   when age_break = "476 and 481" then 104
# MAGIC   when age_break = "481 and 486" then 105
# MAGIC   when age_break = "486 and 491" then 106
# MAGIC   when age_break = "491 and 496" then 107
# MAGIC   when age_break = "496 and 500" then 108
# MAGIC   when age_break = "501 and 506" then 109
# MAGIC   when age_break = "506 and 511" then 110
# MAGIC   when age_break = "511 and 516" then 111
# MAGIC   when age_break = "516 and 521" then 112
# MAGIC   when age_break = "521 and 526" then 113
# MAGIC   when age_break = "526 and 531" then 114
# MAGIC   when age_break = "531 and 536" then 115
# MAGIC   when age_break = "536 and 541" then 116
# MAGIC   when age_break = "541 and 546" then 117
# MAGIC   when age_break = "546 and 551" then 118
# MAGIC   when age_break = "551 and 556" then 119
# MAGIC   when age_break = "556 and 561" then 120
# MAGIC   when age_break = "561 and 566" then 121
# MAGIC   when age_break = "566 and 571" then 122
# MAGIC   when age_break = "571 and 576" then 123
# MAGIC   when age_break = "576 and 581" then 124
# MAGIC   else 125
# MAGIC end

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC     device_age_days,
# MAGIC     sum(is_fraud) as total_is_fraud,
# MAGIC     count(distinct transaction_id) as total_transactions,
# MAGIC     100 * total_is_fraud / total_transactions as fraud_rating
# MAGIC from transactions_without_high_risk
# MAGIC group by 1

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Hour of the day

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC with timestamp_formated as(
# MAGIC   select
# MAGIC     	transaction_id,
# MAGIC     	is_fraud,
# MAGIC timestamp_millis(cast(transaction_timestamp as bigint)) as transcation_timestamp_formatted
# MAGIC   from
# MAGIC     transactions_without_high_risk
# MAGIC )
# MAGIC select
# MAGIC   hour(transcation_timestamp_formatted) as ts_hour,
# MAGIC   count(transaction_id) as total_transactions,
# MAGIC   sum(is_fraud) / total_transactions as fraud_rate
# MAGIC from
# MAGIC   timestamp_formated
# MAGIC group by 1
# MAGIC order by 2 desc
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Flags 

# COMMAND ----------

# MAGIC %sql
# MAGIC with remove_flags as(
# MAGIC   select
# MAGIC t.transaction_id,
# MAGIC   	case when tff.transaction_id is not null then 1 else 0 end as is_fraud
# MAGIC   from 
# MAGIC transactions t
# MAGIC   	left join transactions_fraud_feedback tff on t.transaction_id = tff.transaction_id
# MAGIC   where
# MAGIC has_fake_location = true OR is_emulator = true OR has_root_permissions = true OR app_is_tampered = true
# MAGIC )
# MAGIC
# MAGIC
# MAGIC select
# MAGIC   count(case when is_fraud = 1 then transaction_id end) / count(transaction_id) as total_frauds,
# MAGIC   count(case when is_fraud = 0 then transaction_id end) / count(transaction_id) as total_no_frauds
# MAGIC from
# MAGIC   remove_flags

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Accounts per Device

# COMMAND ----------

# MAGIC %sql
# MAGIC with timestamp_formated as (
# MAGIC   select
# MAGIC     *,
# MAGIC     timestamp_millis(cast(transaction_timestamp as bigint)) as transcation_timestamp_formatted
# MAGIC   from
# MAGIC     transactions
# MAGIC ),
# MAGIC devices_ordered_transaction_history as (
# MAGIC   SELECT
# MAGIC     a.account_id,
# MAGIC     a.transaction_id,
# MAGIC     a.transcation_timestamp_formatted,
# MAGIC     a.device_id,
# MAGIC     count(distinct b.account_id) as CountDistinct
# MAGIC   FROM
# MAGIC     timestamp_formated a
# MAGIC     INNER JOIN timestamp_formated b
# MAGIC       ON a.device_id = b.device_id
# MAGIC       and b.transcation_timestamp_formatted <= a.transcation_timestamp_formatted
# MAGIC   GROUP BY
# MAGIC     a.account_id,
# MAGIC     a.transaction_id,
# MAGIC     a.device_id,
# MAGIC     a.transcation_timestamp_formatted
# MAGIC )
# MAGIC select
# MAGIC   CountDistinct,
# MAGIC   count(distinct devs.transaction_id) as total_transactions,
# MAGIC   count(distinct t.transaction_id) as total_frauds,
# MAGIC   100 * total_frauds / coalesce(total_transactions, 1) as fraud_rate
# MAGIC from
# MAGIC   devices_ordered_transaction_history devs
# MAGIC   left join transactions_fraud_feedback t on devs.transaction_id = t.transaction_id
# MAGIC   left join (
# MAGIC     select
# MAGIC       transaction_id,
# MAGIC       client_decision,
# MAGIC       transaction_value
# MAGIC     from
# MAGIC       transactions
# MAGIC   ) t2 on devs.transaction_id = t2.transaction_id
# MAGIC group by 1
# MAGIC order by 2 desc

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Evaluation of Classification Rules

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Current vs New decision flow

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Fraud data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Current

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct t.transaction_id) as total_transactions,
# MAGIC   count(distinct t2.transaction_id) as total_frauds,
# MAGIC   round(sum(case when t2.transaction_id is not null then transaction_value end) * 0.15, 2) as total_fraud_transaction,
# MAGIC   100 * total_frauds / coalesce(total_transactions, 1) as fraud_rate,
# MAGIC   total_fraud_transaction + round(total_frauds * 0.05, 2) as total_fraud_cost
# MAGIC from
# MAGIC   transactions t
# MAGIC   left join transactions_fraud_feedback t2 on t.transaction_id = t2.transaction_id

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### New

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct transaction_id) as count_transactions,
# MAGIC   count(distinct case when is_fraud = 1 and risk_band = "low" then transaction_id end) + count(distinct case when is_fraud = 1 and risk_band = "mid" then transaction_id end) as count_frauds,
# MAGIC   sum(case when is_fraud = 1 and risk_band = "low" then transaction_value end) as total_frauds_low_band,
# MAGIC   count(distinct case when is_fraud = 0 and risk_band = "low" and client_decision = "denied" then transaction_id end) as count_low_band_denied,
# MAGIC   sum(case when is_fraud = 0 and risk_band = "low" and client_decision = "denied" then transaction_value end) as total_low_band_denied,
# MAGIC
# MAGIC
# MAGIC   sum(case when is_fraud = 1 and risk_band = "mid" then transaction_value end) as total_frauds_mid_band,
# MAGIC  
# MAGIC   count(distinct case when is_fraud = 1 and risk_band = "mid" then transaction_id end) as count_frauds_mid_band,
# MAGIC
# MAGIC
# MAGIC   total_frauds_low_band * 0.15 + total_frauds_mid_band * 0.15 as total_fraud,
# MAGIC   100 * count_frauds / coalesce(count_transactions, 1) as fraud_rate,
# MAGIC   total_fraud + round(count_frauds_mid_band * 0.05, 2) as total_fraud_cost,
# MAGIC   total_fraud_cost + ((fraud_rate / 100) * 200 * count_low_band_denied * 0.15)  as total_fraud_cost_realistic_case
# MAGIC
# MAGIC
# MAGIC from
# MAGIC   transactions_with_bands
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 2FA

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Current

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select
# MAGIC   count(distinct t.transaction_id) as total_transactions,
# MAGIC   round(total_transactions * 0.05, 2) as total_2fa_cost
# MAGIC from
# MAGIC   transactions t

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### New

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select
# MAGIC   count(distinct case when risk_band = "mid" then transaction_id end) as count_mid_band,
# MAGIC   round(count_mid_band * 0.05, 2) as total_2fa_cost
# MAGIC from
# MAGIC   transactions_with_bands

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Revenue

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Current

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC select
# MAGIC   count(distinct t.transaction_id) as total_transactions,
# MAGIC   round(sum(case when t2.transaction_id is not null then transaction_value end) * 0.15, 2) as total_fraud_transaction,
# MAGIC   round(sum(case when t.client_decision = "approved" and t2.transaction_id is null then transaction_value end) * 0.15, 2) as total_approved_transaction,
# MAGIC   coalesce(total_approved_transaction, 0) - coalesce(total_fraud_transaction, 0) - coalesce((0.05 * total_transactions), 0) AS total_revenue
# MAGIC from
# MAGIC   transactions t
# MAGIC   left join transactions_fraud_feedback t2 on t.transaction_id = t2.transaction_id

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### New

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   sum(case when is_fraud = 0 and risk_band = "low" and client_decision = "denied" then transaction_value end) as total_low_band_denied,
# MAGIC  
# MAGIC   count(distinct case when risk_band = "mid" then transaction_id end) as count_mid_band,
# MAGIC
# MAGIC
# MAGIC   sum(case when (risk_band = "low") or (risk_band = "mid" and client_decision = "approved") then transaction_value end) as total_approved,
# MAGIC   sum(case when (is_fraud = 1 and risk_band = "low") or (risk_band = "mid" and is_fraud = 1 and client_decision = "approved") then transaction_value end) as total_fraud_approved,
# MAGIC
# MAGIC
# MAGIC   sum(case when (risk_band = "mid" and is_fraud = 1 and client_decision = "denied") then transaction_value end) as total_fraud_denied,
# MAGIC
# MAGIC
# MAGIC   round(coalesce(total_approved, 0) * 0.15, 2) - round(coalesce(total_fraud_approved, 0) * 0.15, 2) - round(coalesce(total_fraud_denied, 0) * 0.15, 2) - round(coalesce(count_mid_band, 0) * 0.05, 2) as total_revenue,
# MAGIC   round(coalesce(total_low_band_denied, 0) * 0.15, 2) as total_approved_cost_if_all_fraudsters,
# MAGIC   total_revenue - total_approved_cost_if_all_fraudsters as worst_case_scenario
# MAGIC from
# MAGIC   transactions_with_bands
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Transactional approval rate

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Current

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct t.transaction_id) as total_transactions,
# MAGIC   count(distinct case when t.client_decision = "approved" then t.transaction_id end) as total_approvals,
# MAGIC   total_approvals / total_transactions as total_approval_rate
# MAGIC  
# MAGIC from
# MAGIC   transactions t
# MAGIC   left join transactions_fraud_feedback t2 on t.transaction_id = t2.transaction_id

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### New

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct transaction_id) as count_transactions,
# MAGIC   count(distinct case when (risk_band = "low") or (risk_band = "mid" and client_decision = "approved") then transaction_id end) as count_approved,
# MAGIC   count_approved / count_transactions as total_approved
# MAGIC from
# MAGIC   transactions_with_bands

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### False Positives

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Current

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct t.transaction_id) as total_transactions,
# MAGIC   0 as hard_false_positives,
# MAGIC   hard_false_positives / total_transactions as hard_false_positive_rate,
# MAGIC   count(distinct case when t.client_decision = "approved" and t2.transaction_id is null then t.transaction_id end) as soft_false_positives,
# MAGIC   soft_false_positives / total_transactions as soft_false_positive_rate
# MAGIC  
# MAGIC from
# MAGIC   transactions t
# MAGIC   left join transactions_fraud_feedback t2 on t.transaction_id = t2.transaction_id

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### New

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct transaction_id) as total_transactions,
# MAGIC   count(distinct case when is_fraud = 0 and risk_band = "high" and client_decision = "approved" then transaction_id end) as hard_false_positives,
# MAGIC   count(distinct case when is_fraud = 0 and risk_band = "mid" and client_decision = "approved" then transaction_id end) as soft_false_positives,
# MAGIC   hard_false_positives / total_transactions as hard_false_positive_rate,
# MAGIC   soft_false_positives / total_transactions as soft_false_positive_rate
# MAGIC
# MAGIC from transactions_with_bands

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### False Negatives

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Current

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct t.transaction_id) as total_transactions,
# MAGIC   0 as hard_false_negatives,
# MAGIC   hard_false_negatives / total_transactions as hard_false_negative_rate,
# MAGIC   count(distinct case when t.client_decision = "approved" and t2.transaction_id is not null then t.transaction_id end) as soft_false_negative,
# MAGIC   soft_false_negative / total_transactions as soft_false_negative_rate
# MAGIC  
# MAGIC from
# MAGIC   transactions t
# MAGIC   left join transactions_fraud_feedback t2 on t.transaction_id = t2.transaction_id

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### New

# COMMAND ----------

# MAGIC %sql
# MAGIC select
# MAGIC   count(distinct transaction_id) as total_transactions,
# MAGIC   count(distinct case when is_fraud = 1 and risk_band = "low" then transaction_id end) as hard_false_negative,
# MAGIC   count(distinct case when is_fraud = 1 and risk_band = "mid" and client_decision = "approved" then transaction_id end) as soft_false_negative,
# MAGIC   hard_false_negative / total_transactions as hard_false_negative_rate,
# MAGIC   soft_false_negative / total_transactions as soft_false_negative_rate
# MAGIC
# MAGIC
# MAGIC from transactions_with_bands
