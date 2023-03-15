from psutil import virtual_memory
from pyspark import SparkConf
from pyspark.sql import SparkSession
import os
import logger

spark_logger = logger.get_logger(__name__)


def get_spark_session():
    """
    Create a SparkSession with optimized memory and partitions.

    Returns:
        A SparkSession object
    """
    try:
        mem = virtual_memory()
        memory = f'{int(round((mem.total / 2) / 1024 / 1024 / 1024, 0))}G'

        spark_logger.info(f"Memory available: {memory}")

        conf = SparkConf() \
            .set('spark.driver.memory', memory) \
            .set('spark.sql.shuffle.partitions', str(os.cpu_count() * 2)) \
            .set("spark.sql.autoBroadcastJoinThreshold", 50 * 1024 * 1024) \
            .set("spark.sql.debug.maxToStringFields", 50) \
            .setAppName("IForest feature importance")

        spark = SparkSession.builder.config(conf=conf).getOrCreate()
        spark_logger.info("SparkSession created successfully.")
        return spark

    except Exception as e:
        spark_logger.error(f"An error occurred: {e}")
        raise e
