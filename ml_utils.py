from pyspark.ml.tuning import ParamGridBuilder
from configs.tpg_config import TPGConfig
from typing import List, Dict, Any
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import numpy as np
from pyspark.sql.functions import col, when
import pyspark.sql.functions as F
from pyspark.ml.tuning import CrossValidator
from pyspark.sql import DataFrame
import numpy as np
import pandas as pd
import os 
import logger

ml_logger = logger.get_logger(__name__)

def get_search_space(search_space_config: Dict[str, int], start: int = 1, stop: int = 10, num: int = 5) -> np.ndarray:
    """
    Generate a grid of numbers between `start` and `stop` with `num` points, and override the defaults
    with values from the `search_space` dictionary if present.

    Args:
        search_space (dict): A dictionary of parameters to override the defaults.
        start (int, optional): The starting value of the grid. Defaults to 1.
        stop (int, optional): The stopping value of the grid. Defaults to 10.
        num (int, optional): The number of points in the grid. Defaults to 5.

    Returns:
        np.ndarray: A 1D numpy array of values within the specified range and number of points.
    """
    start = search_space_config.get('start', start)
    stop = search_space_config.get('stop', stop)
    num = search_space_config.get('num', num)
    search_space_arr = np.linspace(start=start, stop=stop, num=num)
    return np.around(search_space_arr).astype(int)

def get_param_grid(config: TPGConfig, model: Any) -> List[Dict[str, any]]:
    """
    Generate a grid of hyperparameters for tuning a PySpark model.

    Args:
        config (dict): A dictionary of configuration parameters for the hyperparameter grid.
        model (RandomForestClassifier): The model object of RandomForestClassifier.

    Returns:
        list: A list of dictionaries containing hyperparameters to test.
    """

    try:
        paramGrid = ParamGridBuilder()
        for key, search_space_config in config.get('hp_tuning', {}).items():
            search_space = get_search_space(search_space_config)
            if hasattr(model, key):
                model_parameter = getattr(model, key)
            else:
                raise ValueError(f"Invalid key: {key}")
            paramGrid = paramGrid.addGrid(model_parameter, search_space)
        return paramGrid.build()
    
    except Exception as e:
        ml_logger.error(f"An error occurred while getting RF attribute: {e}")
        raise e


def prepare_data(
        train_data: DataFrame, 
        test_data: DataFrame, 
        featuresCols: List[str], 
        label_col: str, 
        label_indexer_model: Any
    ) -> tuple:
    """
    Prepare training and test data for feature selection.

    Args:
        train_data (Any): Training data as a Spark DataFrame.
        test_data (Any): Test data as a Spark DataFrame.
        featuresCols (List[str]): List of feature column names.
        label_col (str): Name of the label column.
        label_indexer_model (Any): StringIndexerModel for encoding the label column.

    Returns:
        tuple: A tuple of the transformed training and test data.
    """
    train = train_data.select(featuresCols + [label_col])
    test = test_data.select(featuresCols + [label_col])

    assembler = VectorAssembler(inputCols=featuresCols, outputCol="features")

    train = label_indexer_model.transform(train)
    test = label_indexer_model.transform(test)
    train = assembler.transform(train)
    test = assembler.transform(test)

    return train, test


def recursive_feature_elimination(
        train_data: DataFrame, 
        test_data: DataFrame, 
        features_cols: List[str], 
        label_col: str, 
        label_indexer_model: Any, 
        min_feature_count: int
    ) -> List[str]:
    """
    Perform recursive feature elimination on the given training and test data.

    Args:
        train_data (Any): Training data as a Spark DataFrame.
        test_data (Any): Test data as a Spark DataFrame.
        features_cols (List[str]): List of feature column names.
        label_col (str): Name of the label column.
        label_indexer_model (Any): StringIndexerModel for encoding the label column.
        min_feature_count (int): The minimum number of features to select.

    Returns:
        List[str]: List of selected feature column names.
    """
    try:
        selection_flag = True if len(features_cols) >= min_feature_count else False
    except Exception as e:
        ml_logger.error(f"An error occurred while defining minimum feature count: {e}")
        raise e

    while len(features_cols) > min_feature_count and selection_flag:
        train, test = prepare_data(train_data, test_data, features_cols, label_col, label_indexer_model)

        rf = RandomForestClassifier(featuresCol="features", labelCol='label', seed=2023)
        rfModel = rf.fit(train)

        indices = np.where(rfModel.featureImportances.toArray() == 0)[0]

        temp_features = []
        for i in range(len(features_cols)):
            if i not in indices:
                temp_features.append(features_cols[i])

        if len(temp_features) == len(features_cols):
            selection_flag = False
        features_cols = temp_features
    ml_logger.info(f"Feature selection done, {len(features_cols)} features selected.")
    return features_cols


def store_selected_features(
        selected_features: List[str],
        store_path: str = "feature_selection",
        file_name: str = "selected_features"
        ) -> None:
    """
    Store the selected features in a parquet file.

    Args:
        selected_features (List[str]): List of selected feature column names.
        store_path (str): The path to store selected features
        file_name (str): The name of file.

    Returns:
        None
    """
    features_df = pd.DataFrame(selected_features, columns=['selected_features'])

    if not os.path.exists(store_path):
        os.makedirs(store_path)

    features_df.to_parquet(f"{store_path}/{file_name}.parquet")
    ml_logger.info(f"Selected feature stored at {store_path}/{file_name}.parquet.")


def load_selected_features(selected_feature_path: str) -> List[str]:
    """
    Load selected features from the given parquet file.

    Args: 
        selected_feature_path: The path of the parquet file containing selected features.
    Returns:
        A list of selected features.
    """
    df = pd.read_parquet(selected_feature_path)
    selected_features = df['selected_features'].to_list()
    ml_logger.info(f"Selected features found in {selected_feature_path}, {len(selected_features)} features were selected.")
    return selected_features