from pyspark.sql import DataFrame
from pyspark.sql.functions import isnan, when, col, mean
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
import logger

prep_logger = logger.get_logger(__name__)

def cast_columns_to_int(data: DataFrame, cols_list: list) -> DataFrame:
    """
    Cast columns to integer type in a PySpark DataFrame.

    Args:
        data: A PySpark DataFrame
        cols_list: A list of column names
    Returns:
        A PySpark DataFrame with given columns cast to integer
    """
    try:
        for col in cols_list:
            data = data.withColumn(col, data[col].cast(IntegerType()))
        return data

    except Exception as e:
        prep_logger.error(f"An error occurred while casting string columns to int: {e}")
        raise e


def handle_nan_values(
        data: DataFrame, 
        cols_list: list,
        drop_threshold: float = 0.5
    ) -> DataFrame:
    """
    Handle NaN values in a PySpark DataFrame.     

    Args:
        data: A PySpark DataFrame
        cols_list: A list of column names with nan values
        drop_threshold (optional): The threshold to drop the column

    Returns:
        A PySpark DataFrame with NaN values handled
    """
    try:
        nan_counts = data.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in data[cols_list].columns])

        total_rows = data.count()

        for col in nan_counts.columns:
            if nan_counts.collect()[0][col] > drop_threshold * total_rows:
                data = data.drop(col)
            else:
                median_value = data.approxQuantile(col, [0.5], 0.01)[0]
                data = data.fillna(median_value, subset=[col])

        return data

    except Exception as e:
        prep_logger.error(f"An error occurred while handling NaN values: {e}")
        raise e


def rename_columns(data: DataFrame, column_names: list) -> DataFrame:
    """
    Rename columns of a PySpark DataFrame.

    Args:
        data: A PySpark DataFrame
        column_names: A list of new column names
    Returns:
        A PySpark DataFrame with renamed columns
    """
    for i in range(len(column_names)):
        data = data.withColumnRenamed("_c" + str(i), column_names[i])
    return data


def create_default_column_names(num_columns: int) -> list:
    """
    Create default column names for DataFrame.
    This method assumes that the last column is the target variable to predict.

    Args:
        num_columns: The number of columns in the DataFrame
    
    Returns:
        A list of column names
    """
    column_names = [f"feature_{i}" for i in range(1, num_columns)]
    column_names.append("target")
    return column_names
