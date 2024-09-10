import os, shutil
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from database_client import DB_Client
from data_cleaner import DataCleaner
from user_satisfaction import UserStatisfactionCalculator


def prep_data(data: pd.DataFrame):
    """
    Prepares the input data by cleaning missing values and dropping unnecessary records.

    This function initializes a `DataCleaner` object to fill missing values in both categorical and numeric 
    columns of interest. It uses the mode for categorical columns and the mean for numeric columns. After 
    handling missing values, the function drops any remaining rows with missing data and returns the cleaned DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame that requires data cleaning.

    Returns:
        pd.DataFrame: The cleaned DataFrame with missing values filled and unnecessary rows dropped.
    """

    # initailize the DataCleaner
    cleaner = DataCleaner(data=data)
    print("########## Databace Cleaner Initialized ##########")

    # define the columns of interest
    columns_of_interest = ["MSISDN/Number", "Avg RTT DL (ms)", "Avg RTT UL (ms)", "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "Handset Type", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]

    # define columns where we will use mode to replace the NA values
    mode_columns = ["MSISDN/Number", "Handset Type"]

    # define columns where we will use mean to replace the NA values
    mean_columns = [col for col in columns_of_interest if col not in mode_columns]

    # clean the categorical data(ones who use mode for their NA)
    data[mode_columns] = cleaner.fill_na(columns=mode_columns, method='mode')

    # clean the numeric data(ones who use mean for their NA)
    data[mean_columns] = cleaner.fill_na(columns=mean_columns, method='mean')

    # clean the engagement data
    columns_of_interest = ["Start", "Start ms", "End", "End ms", "Dur. (ms)", "Dur. (ms).1"]

    ## this needs change
    data = cleaner.drop_na()
    
    return data

def transform_data(data: pd.DataFrame):
    """
    Transforms the input data by calculating engagement, experience, and satisfaction scores, and returns 
    the feature set and target variable for model training.

    This function uses the `UserStatisfactionCalculator` to calculate clusters and scores for engagement 
    and experience. It also computes the satisfaction score for each user. The transformed data is then 
    filtered to include only the necessary columns, and split into feature variables (X) and target variables (y).

    Args:
        data (pd.DataFrame): The input DataFrame containing the user data.

    Returns:
        tuple: A tuple containing:
            - X (pd.DataFrame): The feature set (engagement and experience scores).
            - y (pd.DataFrame): The target variable (satisfaction score).
    """

    # Assign satisfcation scores
    satisfaction_calc = UserStatisfactionCalculator(data=data)

    # obtain the experience clusters
    experience_clusters = satisfaction_calc.get_experience_cluster()
    experience_score = satisfaction_calc.claculate_experience_score()

    # obtain the experience clusters
    engagement_clusters = satisfaction_calc.get_engagement_cluster()
    engagement_score = satisfaction_calc.calculate_engagement_score()

    # calculate the satisfaction scores of the users
    satisfaction_scores = satisfaction_calc.get_satifisfaction_score(engagemet_score=engagement_score['engagement_score'], experience_score=experience_score['experience_score'])

    # Add the metrics to a new dataframe
    engagement_score['experience_score'] = experience_score['experience_score']
    engagement_score['satisfaction_score'] = satisfaction_scores

    unwanted_cols = [col for col in engagement_score.columns if col not in ['experience_score', 'satisfaction_score', 'engagement_score']]
    user_data = engagement_score.drop(columns=unwanted_cols)

    # obtain the features and target variables
    X = user_data.drop(columns=['satisfaction_score'])
    y = user_data.drop(columns=['engagement_score', 'experience_score'])

    return (X, y)

def initialize_mlflow(uri: str, experiment_name: str):
    """
    Initializes MLflow by setting the tracking URI and retrieving or creating an experiment.

    This function sets the MLflow tracking URI, either creates a new experiment with the provided name,
    or retrieves the existing experiment if it already exists. The function returns the experiment ID.

    Args:
        uri (str): The tracking URI for MLflow (e.g., the location to log runs and artifacts).
        experiment_name (str): The name of the experiment to create or retrieve.

    Returns:
        str: The ID of the experiment.
    """

    # Initialize MLflow
    mlflow.set_tracking_uri("mlruns")  # Set up local directory for logging (In this case a directory called test in the same folder as the script)
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        search_result = mlflow.search_experiments(filter_string=f"name = '{experiment_name}'")
        experiment_id = search_result[0].experiment_id

    return experiment_id

def train_and_log_model(model, model_name, experiment_id):
    """
    Trains a given machine learning model, logs its performance metrics and parameters to an MLflow experiment,
    and saves the trained model.

    This function starts an MLflow run within the specified experiment, trains the provided model on the training data,
    evaluates it on the test data by calculating the mean squared error (MSE), and logs both the model and the metrics.

    Args:
        model (object): The machine learning model to be trained (e.g., sklearn model).
        model_name (str): The name of the model, used as the run name in MLflow and for logging.
        experiment_id (str or int): The ID of the MLflow experiment where the run should be logged.

    Returns:
        float: The mean squared error (MSE) of the model on the test data.
    """

    with mlflow.start_run(run_name=model_name, experiment_id=experiment_id):
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and calculate performance
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        
        # Log the model and parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} - MSE: {mse:.4f}")
        return mse

def log_best_model(experiment_name: str, tracking_uri: str):
    """
    Logs the best model from an MLflow experiment based on the lowest mean squared error (mse) metric.
    
    This function searches for completed runs in the specified experiment, ranks them by their 
    mean squared error (mse), and identifies the best model. It retrieves the best model's 
    artifact URI, logs its associated metrics, and returns the model's path and metrics as a dataframe.

    Args:
        experiment_name (str): The name of the MLflow experiment to search for the best model.
        tracking_uri (str): The tracking URI of the MLflow server where the experiment runs are logged.

    Returns:
        dict: A dictionary containing the following:
            - "metrics" (pandas.DataFrame): A DataFrame containing the metrics (mse) and model type.
            - "path" (str): The relative path to the best model's artifacts.
    """
    # find the best model
    run_name = 'Neural_Net_Training'
    runs = mlflow.search_runs(experiment_names=[experiment_name] , filter_string=f'''attributes.status = "FINISHED"''' , order_by=["metrics.mse DESC"])        
    best_run = runs.head(n=1)
    # get the artifacts uri
    artifact_uri = best_run["artifact_uri"][0]
    # get the metrics
    for col in best_run.columns:
            if 'metrics' in col:
                    name = col.split('.')[1]
                    value = best_run[col][0]
                    print(f"{name} = {value}")       
    
    model_type = best_run["params.model_name"][0]
    metrics = pd.DataFrame(data={
            "name" : [name],
            "value": [value],
            "model_type": [model_type]
    })
    # make the path relative
    path = ''.join([tracking_uri, artifact_uri.split(tracking_uri)[1]]) + f"/{model_type}"
    return {
        "metrics" : metrics,
        "path": path  
    }

def save_model(path: str, metrics: pd.DataFrame):
    """
    Saves the best model from a specified path and metrics to a designated folder.
    Removes any existing 'model.pkl' file before saving the new one.
    
    Args:
        path (str): The directory containing the model file (e.g., path to MLflow run folder).
        metrics (pd.DataFrame): The metrics dataframe to be saved alongside the model.
    """
    # The folder path to store the best model
    save_path = './models/model_object'
    
    # Ensure the save_path exists
    os.makedirs(save_path, exist_ok=True)
    
    # Define the save path for the model and metrics
    model_save_path = os.path.join(save_path, 'model.pkl')
    
    # Define the path of the model from MLflow or any other source
    model_path = os.path.join(path, 'model.pkl')

    try:
        # If 'model.pkl' already exists, remove it before saving the new model
        if os.path.exists(model_save_path):
            os.remove(model_save_path)
            print(f"Existing model file '{model_save_path}' removed.")

        # Copy the new model from model_path to the save_path
        shutil.copyfile(model_path, model_save_path)
        print(f"Model saved successfully at {model_save_path}")

        # Save the metrics to CSV in the './models' folder, replacing any existing file
        metrics.to_csv(path_or_buf='./models/metrics.csv', index=False)
        print("Metrics saved successfully at './models/metrics.csv'")
    
    except Exception as e:
        print(f"Failed to save model or metrics: {e}")


if __name__ == '__main__':
    # values you can change
    experiment_name = 'Regression Models Experiment'
    tracking_uri = 'mlrun'

    # obtain values form environment variables
    host = os.getenv("DB_HOST")
    user_name = os.getenv("DB_USER")
    passowrd = os.getenv("DB_PASSWORD")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")

    # initialize the DB_Client
    db_client = DB_Client(
        host=host,
        user_name=user_name,
        password=passowrd,
        port=port,
        database_name=database
    )
    print("########## Databace Client Initialized ##########")

    # load the data
    data = db_client.dump_data()
    print("########## Data Loaded from Database ##########")

    # clean and prep the data
    data = prep_data(data=data)
    print("########## Data Cleaned ##########")
    
    # transfrom the data
    X, y = transform_data(data=data)
    print("########## Data Transformation Finished ##########")

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("########## Data Spliting Finished ##########")
    
    # Initialize MLflow
    experiment_id = initialize_mlflow(uri=tracking_uri, experiment_name=experiment_name)
    print("########## MLflow tracker initialized ##########")

    # Train and log Linear Regression
    lr_model = LinearRegression()
    lr_mse = train_and_log_model(lr_model, "LinearRegression", experiment_id=experiment_id)
    print("########## Linear Regression Training Finished ##########")

    # Train and log RandomForest Regressor
    rf_model = RandomForestRegressor(random_state=42)
    rf_mse = train_and_log_model(rf_model, "RandomForestRegressor", experiment_id=experiment_id)
    print("########## Random Forest Regressor Training Finished ##########")

    # Train and log Support Vector Regressor
    svr_model = SVR()
    svr_mse = train_and_log_model(svr_model, "SupportVectorRegressor", experiment_id=experiment_id)
    print("########## MLflow tracker initialized ##########")
    
    # log the best model
    best_model = log_best_model(experiment_name=experiment_name, tracking_uri=tracking_uri)    
    save_model(path=best_model['path'], metrics=best_model['metrics'])
    print("########## Best Model Logged ##########")

