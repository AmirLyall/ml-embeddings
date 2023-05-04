# Databricks notebook source
# %pip install pymilvus milvus

# COMMAND ----------

# MAGIC %md #Establish Connection to Milvus Server

# COMMAND ----------

import numpy as np
from pyspark.sql.functions import col,lit
from pyspark.sql.types import StringType
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection)

connections.connect(
  alias="default",
  uri="tcp://ec2-18-237-71-2.us-west-2.compute.amazonaws.com:19530"
)

# COMMAND ----------

# MAGIC %md #Define and Create a Collection

# COMMAND ----------

def create_collection():
    field1 = FieldSchema(name=_ID_FIELD_NAME, 
                         dtype=DataType.INT64, 
                         description="int64", is_primary=True)
    field2 = FieldSchema(name=_VECTOR_FIELD_NAME, 
                         dtype=DataType.FLOAT_VECTOR, 
                         description="float vector", dim=_DIM,
                         is_primary=False)
    field3 = FieldSchema(name=_STR_FIELD_NAME, 
                         dtype=DataType.VARCHAR, 
                         description="string",
                         max_length=_MAX_LENGTH, 
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2, field3], 
                              description=_COLLECTION_DESCRIPTION)
    collection = Collection(name=_COLLECTION_NAME, 
                            data=None, 
                            schema=schema)
    print("\nCollection created:", _COLLECTION_NAME)
    return collection

# COMMAND ----------

_COLLECTION_NAME = 'product_combined_embeddings'
_COLLECTION_DESCRIPTION = "image and text embeddings for a collection of products - reference dbsql table:sfgs.all_embeddings"
_ID_FIELD_NAME = 'id'
_VECTOR_FIELD_NAME = 'combined_embedding'
_STR_FIELD_NAME = "path"
_DIM = 1536
_MAX_LENGTH = 1024

# COMMAND ----------

image_text_df = spark.sql("select id, description, product, path_cleaned as path, concat(embedding, text_embedding) as combined_embedding from sgfs.all_embeddings").toPandas()

#create image collection
product_image_text_collection=create_collection()
#select relevant columns
# write_image_text_df = image_text_df[['id','combined_embedding','path']]
#write to milvus collection
# insert_result = product_image_text_collection.insert(write_image_text_df)

# COMMAND ----------

from pymilvus import utility
utility.list_collections()

# COMMAND ----------

# MAGIC %md #Load Collection

# COMMAND ----------

collection = Collection("product_combined_embeddings")

# COMMAND ----------

# index = {
#     "index_type": "IVF_FLAT",
#     "metric_type": "L2",
#     "params": {"nlist": 128},
# }

# product_image_text_collection.create_index(_VECTOR_FIELD_NAME, index)

# COMMAND ----------

from pymilvus import Collection
product_image_text_collection = Collection(name=_COLLECTION_NAME)
product_image_text_collection.load()

# COMMAND ----------

# MAGIC %md #Obtain Top 3 Matches/Weights

# COMMAND ----------

import time
import pandas as pd

#Define Search Vector
vector_to_search = [image_text_df.combined_embedding]

#Loop Through Embeddings
for i in range(len(image_text_df.combined_embedding)):
  #Define Search Parameters for Similarity
  search_params = {
      "metric_type": "L2",
      "params": {"nprobe": 10},
  }

  #Calculate Lookup Time
  start_time = time.time()
  result = product_image_text_collection.search([image_text_df.combined_embedding[i]],_VECTOR_FIELD_NAME, search_params, limit=4)
  end_time = time.time()

  print(f"embedding_id: {i+1}")
  
  #Embedding IDs
  result_ids = list([(a.ids) for a in result][0])[1:4]
  print(f"top_3_similar_embedding_ids: {result_ids}")

  #Obtain Euclidean Distanes (L2)
  top_3_similar_embedding_distances = list([(a.distances) for a in result][0])[1:4]
  print(f"top_3_similar_embedding_distances: {top_3_similar_embedding_distances}")

  #Obtain Weightings Inverse to Distance
  inverse_distances = [1/distance for distance in top_3_similar_embedding_distances]
  top_3_similar_embedding_weights = [inverse_distance/(sum(inverse_distances)) for inverse_distance in inverse_distances]
  print(f"top_3_similar_embedding_weights: {top_3_similar_embedding_weights}")

  #Print Lookup Time
  search_latency_fmt = "search latency = {:.4f}s"
  print(search_latency_fmt.format(end_time - start_time))

  print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# COMMAND ----------


