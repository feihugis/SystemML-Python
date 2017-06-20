__all__ = [ 'sc', 'spark', 'sqlCtx']

import os

os.environ["SPARK_HOME"] = "/Users/fei.hu1@ibm.com/Documents/Software/spark-2.1.1-bin-hadoop2.7"

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.context import SQLContext

sc = SparkContext(appName="SystemML_Learning", master="local[4]")
spark = SparkSession.builder.appName("SystemML_Learning").getOrCreate()
sqlCtx = SQLContext(sc)