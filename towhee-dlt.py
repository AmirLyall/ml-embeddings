# Databricks notebook source
# MAGIC %md ##Install the Required Packages and Dependencies

# COMMAND ----------

# MAGIC %pip install towhee
# MAGIC %pip install opencv-python
# MAGIC %pip install torch>=1.9.0
# MAGIC %pip install torchvision>=0.10
# MAGIC %pip install timm
# MAGIC %pip install transformers>=4.15.0

# COMMAND ----------

import dlt
import towhee
import pandas as pd
from typing import Iterator,Tuple
from pyspark.sql.functions import pandas_udf, col, regexp_replace
from pyspark.sql.types import ArrayType, DoubleType, StringType

# COMMAND ----------

# MAGIC %md ##Define Image Data Source (Bronze)

# COMMAND ----------

@dlt.create_table(comment="Load data from s3 to table")
def source_data():
#   img_folder = "<your image location>"
  return spark.readStream.format("cloudFiles").option("cloudFiles.format", "binaryfile").load(img_folder)

# COMMAND ----------

# MAGIC %md ##Define Pandas UDF for Images

# COMMAND ----------

@pandas_udf(ArrayType(DoubleType()))
def get_blip_embedding_from_img(iterator: Iterator[pd.DataFrame])-> Iterator[pd.Series]:
  import glob
  import towhee
  for features in iterator:
    yield pd.Series(towhee.glob(*features) \
      .image_decode() \
#       incorporate desired model here
      .image_text_embedding.blip(model_name='blip_base', modality='image'))

# COMMAND ----------

# MAGIC %md ##Generate Embeddings for Image Data (Silver)

# COMMAND ----------

@dlt.create_table(comment="extract image embeddings",
                  schema= """
                  path_cleaned STRING,
                  embedding ARRAY<DOUBLE>,
                  id BIGINT GENERATED ALWAYS AS IDENTITY
                  """)
def image_embeddings():
  return (dlt.readStream("source_data")
          .select("path")
          .withColumn('path_cleaned', regexp_replace('path', 'dbfs:', '/dbfs'))
          .select("path_cleaned",)
          .withColumn("embedding",get_blip_embedding_from_img(col("path_cleaned")))
          .select("path_cleaned","embedding")
         )

# COMMAND ----------

# MAGIC %md ##Define Text Data Source (Bronze)

# COMMAND ----------

@dlt.create_table(comment="Load text data from s3 to table")
def source_data_text():
  return spark.sql("select * from <your warehouse>.item_descriptions")

# COMMAND ----------

# MAGIC %md ##Define Pandas UDF for Text

# COMMAND ----------

@pandas_udf(ArrayType(DoubleType()))
def get_blip_embedding_from_text(iterator: Iterator[pd.DataFrame])-> Iterator[pd.Series]:
  import glob
  import towhee
  for features in iterator:
    yield pd.Series(towhee.dc(list(features)) \
      .image_text_embedding.blip(model_name='blip_base', modality='text'))

# COMMAND ----------

# MAGIC %md ##Generate Embeddings for Text Data (Silver)

# COMMAND ----------

@dlt.create_table(comment="extract text embeddings",
                  schema="""
                  description STRING,
                  text_embedding ARRAY<DOUBLE>,
                  product STRING,
                  path_cleaned STRING,
                  id BIGINT GENERATED ALWAYS AS IDENTITY
                  """
                 )
def text_embeddings():
  return (dlt.readStream("source_data_text")
          .withColumn("text_embedding", get_blip_embedding_from_text(col("description")))
          .withColumn('path_cleaned', regexp_replace('dbfs_path', 'dbfs:', '/dbfs'))
          .select("description","text_embedding","product","path_cleaned")
         )

# COMMAND ----------

# MAGIC %md ##Merge the Embeddings (Gold)

# COMMAND ----------

@dlt.create_table(comment="combine all embeddings")
def all_embeddings():
  return (dlt.read("image_embeddings").select("id","embedding")
          .join
          (dlt.read("text_embeddings"), "id", "left")
         )
