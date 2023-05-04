# Databricks notebook source
# MAGIC %md #Forecasting New Products Utilizing Similarity Indices

# COMMAND ----------

# MAGIC %md ##Workflow:
# MAGIC <b>Part A (Previous Blog)</b>: Ingest Image Data --> Generate Embeddings --> Store in Vector Database (Milvus) --> Search/Rank Similar Items
# MAGIC
# MAGIC <b>Part B</b>: Data Preparation: Identify Top Sellers --> Add Sales IDs for Top Sellers to Image Data --> Join Image Data with Sales Data
# MAGIC
# MAGIC <b>Part C</b>: Generate Weighted Demand for Trend and Seasonality --> Scale the Demand Accordingly<br>
# MAGIC
# MAGIC time series per product (dept)<br>
# MAGIC from part A find top three similar products<br>
# MAGIC take corresponding time series for the top 3, rescale between 0 and 1<br>
# MAGIC merge and generate a single time series based on similarity<br>
# MAGIC
# MAGIC example<br>
# MAGIC image 1 - 65%<br>
# MAGIC image 2 - 45%<br>
# MAGIC image 3 - 30%<br>
# MAGIC
# MAGIC 65/sum of above, etc. to determine weighted trends<br>
# MAGIC how to consider magnitude

# COMMAND ----------

# MAGIC %pip install typing-extensions --upgrade

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

# MAGIC %reload_ext autoreload
# MAGIC %autoreload 2
# MAGIC %matplotlib inline
# MAGIC
# MAGIC import pandas as pd
# MAGIC import numpy as np
# MAGIC import matplotlib.pyplot as plt

# COMMAND ----------

# MAGIC %pip install darts

# COMMAND ----------

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.models import ExponentialSmoothing, TBATS, AutoARIMA, Theta
from darts.metrics import smape, mape

# COMMAND ----------

# MAGIC %md ##Part B

# COMMAND ----------

# Create dataframe for image data

img_folder = "/mnt/oetrta/databricks-amir-retail-images"

shoes = spark.read.format("binaryFile").load(img_folder)

shoes.display()

# COMMAND ----------

# Read sales dataset into a dataframe

walmart = spark.read\
          .option("header", True)\
          .option("inferSchema", True)\
          .csv("dbfs:/FileStore/amir.lyall@databricks.com/walmart_cleaned.csv")

walmart = walmart.drop("_c0")

display(walmart)

# COMMAND ----------

# Create a temp view

walmart.createOrReplaceTempView("walmart_tmp")

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC -- Identify top 11 departments by total sales which will be used to mimic shoe sales
# MAGIC
# MAGIC select distinct dept, sum(Weekly_Sales) as total_sales from walmart_tmp
# MAGIC group by dept
# MAGIC order by total_sales desc, dept
# MAGIC limit 11

# COMMAND ----------

# Create dataframe with top 11 departments by total sales

top_sales = spark.sql("select dept, sum(weekly_sales) as total_weekly_sales, date, avg(temperature) as avg_temp from walmart_tmp\
            where dept in (92, 95, 38, 72, 90, 40, 2, 91, 13, 8, 94)\
            group by dept, date\
            order by date")

top_sales.display()

# COMMAND ----------

# Add department IDs to shoes data in no particular order

from pyspark.sql.functions import monotonically_increasing_id, row_number, split
from pyspark.sql import Window

top_dept = [92, 95, 38, 72, 90, 40, 2, 91, 13, 8, 94]
top_dept_df = sqlContext.createDataFrame([(dept_num,) for dept_num in top_dept], ['dept'])

shoes = shoes.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
top_dept_df = top_dept_df.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

shoes_dept = shoes.join(top_dept_df, shoes.row_idx == top_dept_df.row_idx).select((split("path", "/").getItem(4)).alias("ImageName"), "dept")

shoes_dept.display()

# COMMAND ----------

#Join shoes data with sales data

shoes_sales = top_sales.join(shoes_dept, top_sales.dept == shoes_dept.dept, 'left').drop(shoes_dept.dept).withColumnRenamed("dept", "shoe_id").orderBy("date", "total_weekly_sales", "shoe_id")
shoes_sales.display()

# COMMAND ----------

shoes_sales.createOrReplaceTempView("test")

# COMMAND ----------

# MAGIC %md ##Part C

# COMMAND ----------

#need to generate time series per product
#choose an image and find top 3 matches (sathish's program)
#generate the sales for each match, scale, and sum accordingly
#blog should reference an additional step which can scale the demand with known ramp-up data based on marketing, clicks, etc.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###Insert New Image/Sales Data

# COMMAND ----------

df = shoes_sales.toPandas()
df["date"]=pd.to_datetime(df["date"])
df.columns = [i.lower() for i in df.columns]
df.info()

# COMMAND ----------

df.groupby('shoe_id').plot(x='date', y='total_weekly_sales',figsize=(26,10)); #plot top 11 dept sales

# df.set_index('date', inplace=True)
# df.groupby('shoe_id')['total_weekly_sales'].plot(legend=True)

# COMMAND ----------

temp_covariates = TimeSeries.from_group_dataframe(df[['date','shoe_id','avg_temp']],group_cols=["shoe_id"],time_col="date",fill_missing_dates=True,
                                                  freq= 'W-FRI') #temperature covariate

# COMMAND ----------

multi_series_ts = TimeSeries.from_group_dataframe(df[['date','shoe_id','total_weekly_sales']],group_cols=["shoe_id"],time_col="date",fill_missing_dates=True,
                                                  freq= 'W-FRI')  # target series by shoe_id

# COMMAND ----------

multi_series_ts[0].plot()

# COMMAND ----------

#check to see if your covariates are of the same size
for i in range(len(multi_series_ts)):
  assert len(multi_series_ts[i]) == len(temp_covariates[i])

# COMMAND ----------

from darts.dataprocessing.transformers import Scaler
#Perform Standard Scaling
scaler = Scaler()
scaled_series_ts = scaler.fit_transform(multi_series_ts)
scaler_cov = Scaler()
scaled_temp_ts = scaler_cov.fit_transform(temp_covariates)
scaled_series_ts[0].plot()
scaled_temp_ts[0].plot()

# COMMAND ----------

# Test to see if there are short time series
short_time_series=[(i,len(scaled_series_ts[i])) for i in range(len(scaled_series_ts)) if len(scaled_series_ts[i]) <= 52*2]
print(f"There are a total of {len(short_time_series)} which do not meet the criteria")
{(scaled_series_ts[i[0]].plot(), print(f"TimeSeries Index {i[0]} has only {i[1]} records")) for i in short_time_series if len(short_time_series)>0};

# COMMAND ----------

from darts.models import NBEATSModel
from time import perf_counter

model = NBEATSModel(
    input_chunk_length=24,
    output_chunk_length=12,
    random_state=42,
    pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]},
)
start_time = perf_counter()
# model.fit(scaled_series_ts[:-36], past_covariates=scaled_temp_ts[:-36], epochs=20, verbose=True)
model.fit(scaled_series_ts, past_covariates=scaled_temp_ts, epochs=20, verbose=True)
end_time = perf_counter()
elapsed_time = end_time - start_time

# COMMAND ----------

print(f"Total Cost: {round(elapsed_time/len(scaled_temp_ts),2)} secs per time series")

# COMMAND ----------

# MAGIC %md #Generate Forecasts for Each Product

# COMMAND ----------

#run forecasts for all 11 pids

# COMMAND ----------

from darts.metrics import mape
# for i in result_ids:
print(f"INFO: Processing TS No: {i}")
preds = model.predict(n=36,series=scaled_series_ts[i][:-36], past_covariates=scaled_temp_ts[i])
actuals = scaler.inverse_transform(scaled_series_ts[i])
plt.figure(figsize=(8, 6))
actuals.plot(label='actual')
preds = scaler.inverse_transform(preds)
preds.plot(label='forecast')
mape(actuals,preds)
print(f"The MAPE for the Time Series is {mape(actuals,preds)}")

# COMMAND ----------

preds

# COMMAND ----------

#all products > products without history (do similarity search and weighting)
#all forecasts
#lookup
#weighting
