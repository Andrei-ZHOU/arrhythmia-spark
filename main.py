import logger
import preprocessing as prep
import spark_utils as s
import ml_utils as ml
from configs.config_loader import load_config
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
import os
import pandas as pd
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics
import json


SELECTED_FEATURE_PATH = "/feature_selection/"
FILE_NAME = "selected_features.parquet"

HP_PATH = "/hyperparameter/"
HP_FILE_NAME = "random_forest_hyperparameters.json"

def model_training_step(config_path: str):
    config = load_config(config_path)

    model_path = f"model_{config['model_name']}"

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    tpg_logger = logger.setup_logger(file_name = model_path+'/tpg_spark_pipeline.log', is_debug = True)

    # Preparation
    spark = s.get_spark_session()

    data = spark.read.csv(config['data_path'], header=False, inferSchema=True)
    tpg_logger.info(f"Data loaded from {config['data_path']}")

    #Rename columns
    num_columns = len(data.columns)
    column_names = prep.create_default_column_names(num_columns)
    data = prep.rename_columns(data, column_names)

    # Preprocessing
    data = data.replace("?", None)
    string_cols = [col[0] for col in data.dtypes if col[1] == 'string']
    data = prep.cast_columns_to_int(data, string_cols)
    data = prep.handle_nan_values(data, string_cols)

    tpg_logger.info("Preprocessing is completed")

    (train_data, test_data) = data.randomSplit([0.8, 0.2], seed=config['random_seed'])

    # define features column
    features_cols = data.columns[:-1]
    label_col = data.columns[-1]

    try:
        label_indexer = StringIndexer(inputCol=label_col, outputCol='label')
        label_indexer_model = label_indexer.fit(data)
    except Exception as e:
        tpg_logger.error(f"An error occurred while creating label indexer: {e}")
        raise e
    
    #Feature Selection
    if os.path.exists(model_path + SELECTED_FEATURE_PATH):
        selected_features = ml.load_selected_features(SELECTED_FEATURE_PATH+FILE_NAME)
    else:
        selected_features = ml.recursive_feature_elimination(
            train_data=train_data,
            test_data=test_data,
            features_cols=features_cols,
            label_col=label_col,
            label_indexer_model=label_indexer_model,
            min_feature_count=config['min_feature_count']
        )
        ml.store_selected_features(
            selected_features=selected_features,
            store_path=model_path+SELECTED_FEATURE_PATH,
            file_name=FILE_NAME       
            )
        
    train, test = ml.prepare_data(train_data, test_data, features_cols, label_col, label_indexer_model)
    rf = RandomForestClassifier(featuresCol = "features", labelCol = 'label', seed=config['random_seed'])

    #Hyperparameter Tuning
    if os.path.exists(model_path + HP_PATH + HP_FILE_NAME):
        with open(model_path + HP_PATH + HP_FILE_NAME, "r") as f:
            hyperparameters = json.load(f)

        # Set the hyperparameters from the JSON file
        for param_name, value in hyperparameters.items():
            param = rf.getParam(param_name)
            rf.set(param, value)

        model = rf.fit(train)

    else:
        param_grid = ml.get_param_grid(config, rf)
        crossval = CrossValidator(estimator=rf,
                                estimatorParamMaps=param_grid,
                                evaluator=MulticlassClassificationEvaluator(),
                                numFolds=5)
        cvModel = crossval.fit(train)

        param_map = cvModel.bestModel.extractParamMap()
        param_map_dict = {str(param.name): value for param, value in param_map.items()}
        import json

        if not os.path.exists(model_path+HP_PATH):
            os.makedirs(model_path+HP_PATH)

        # Save the dictionary as a JSON file
        with open(model_path + HP_PATH + HP_FILE_NAME, "w") as f:
            json.dump(param_map_dict, f)

        model = cvModel.bestModel

    predictions = model.transform(test)

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")
    accuracy = evaluator.evaluate(predictions.select("label", "prediction"))
    tpg_logger.info(f"Model is evaluated, the accuracy is {accuracy}.")


    metrics = MulticlassMetrics(predictions.select("label", "prediction").rdd.map(tuple))

        # Confusion Matrix
    tpg_logger.info(f"Model is evaluated, the confusion matrix is {metrics.confusionMatrix().toArray()}.")

    model.save(model_path+"/final_model")
    tpg_logger.info(f"Model saved to path {model_path}/final_model.")
    tpg_logger.info(f"Modelling pipeline for {model_path} is completed.")


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config', help='The path to the config file for this pipeline.'
    )
    args = parser.parse_args()
    model_training_step(args.config)
