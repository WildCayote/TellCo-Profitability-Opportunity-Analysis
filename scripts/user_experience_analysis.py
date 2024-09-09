import math, os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from typing import List

from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

from database_client import DB_Client
from data_cleaner import DataCleaner

def print_key_value(data: pd.DataFrame):
    '''
    A function that will print 
    '''

def aggregate_experience_information(cleaner: DataCleaner, columns_of_interest: List[str], mode_columns: List[str], mean_columns: List[str]):
    '''
    A function that encapsulates the logic for aggregating user experience information.

    Args:
        cleaner (DataCleaner): an instance of a DataCleaner class
        columns_of_interest (List(str)): the list of names of the columns you want to work with
        mode_columns (List(str)): the list of names of columns we want to replace missing values with the mode of the respective column
        mean_columns (List(str)): the list of names of columns we want to replace missing values with the mean of the respective column
    
    Returns:
        pd.Dataframe: a dataframe that has users MSISDN/Number as the key and the experience columns.
    '''

    # obtain the data from the cleaner
    data = cleaner.data

    # clean the categorical data(ones who use mode for their NA)
    data[mode_columns] = cleaner.fill_na(columns=mode_columns, method='mode')
    
    # clean the numeric data(ones who use mean for their NA)
    data[mean_columns] = cleaner.fill_na(columns=mean_columns, method='mean')
    
    print("---------- Databace Client Initialized ----------")

    # Group the customers using MSISDN/Number
    customer_grouping = data.groupby(by="MSISDN/Number")

    # aggregate the numeric values using average
    avg_stats = customer_grouping.agg({
        "Avg RTT DL (ms)": "mean",
        "Avg RTT UL (ms)": "mean",
        "TCP DL Retrans. Vol (Bytes)": "mean",
        "TCP UL Retrans. Vol (Bytes)": "mean",
        "Avg Bearer TP DL (kbps)": "mean",
        "Avg Bearer TP UL (kbps)": "mean",
        "Handset Type": lambda x: x.mode().iloc[0] # find the most frequent handset type for the user
    })

    # add the respective averages
    avg_stats["avg_rtt"] = avg_stats[["Avg RTT DL (ms)" , "Avg RTT UL (ms)"]].mean(axis=1)
    avg_stats["avg_tcp_rt"] = avg_stats[["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)"]].mean(axis=1)
    avg_stats["avg_throughput"] = avg_stats[["Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]].mean(axis=1)

    # rename column Handset Type
    avg_stats = avg_stats.rename(columns={
        "Handset Type": "handset_type"
    })

    # drop the old columns
    avg_stats = avg_stats.drop(columns=["Avg RTT DL (ms)" , "Avg RTT UL (ms)", "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"])

    return avg_stats

def compute_top_experience_metrics(user_experience_cleaned: pd.DataFrame):
    '''
    A function that computes the 10 top, bottom and most frequent values in a cleaned and aggregated user experience data.

    Args:
        user_experience_cleaned (pd.DataFrame): a dataframe of cleaned user experience information
    
    Returns:
        dict: a dictionary of arrays which contain 3 dataframes each corresponding to top 10, bottom 10 and most frequent 10 for the respective column
    '''

    # sort the data using average tcp 
    tcp_sorted = user_experience_cleaned.sort_values(by="avg_tcp_rt", ascending=False)

    # sort the data using average tcp 
    rtt_sorted = user_experience_cleaned.sort_values(by="avg_rtt", ascending=False)

    # sort the data using average tcp 
    thoughput_sorted = user_experience_cleaned.sort_values(by="avg_throughput", ascending=False)

    # compile the result
    result = {
        "TCP_RT": [
            tcp_sorted.head(10)['avg_tcp_rt'],
            tcp_sorted.tail(10)['avg_tcp_rt'],
            user_experience_cleaned["avg_rtt"].value_counts().head(10)   
        ],
        "RTT": [
           rtt_sorted.head(10)['avg_rtt'],
           rtt_sorted.tail(10)['avg_rtt'],
           user_experience_cleaned["avg_rtt"].value_counts().head(10)    
        ],
        "Throughput": [
          thoughput_sorted.head(10)['avg_throughput'],
          thoughput_sorted.tail(10)['avg_throughput'],
          user_experience_cleaned["avg_throughput"].value_counts().head(10)     
        ]
    }

    return result

def compute_handset_average_metrics():
    ''''''


def cluster_users():
    ''''''



if __name__ == '__main__':
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

    # initailize the DataCleaner
    cleaner = DataCleaner(data=data)
    print("########## Databace Cleaner Initialized ##########")

    # define the columns of interest
    columns_of_interest = ["MSISDN/Number", "Avg RTT DL (ms)", "Avg RTT UL (ms)", "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "Handset Type", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]

    # now let us define columns where we will use mode to replace the NA values
    mode_columns = ["MSISDN/Number", "Handset Type"]

    # now let us define columns where we will use mean to replace the NA values
    mean_columns = [col for col in columns_of_interest if col not in mode_columns]

    '''Task 3.1'''
    user_experience = aggregate_experience_information(cleaner=cleaner, columns_of_interest=columns_of_interest, mode_columns=mode_columns, mean_columns=mean_columns)
    print(user_experience)

    '''Task 3.2'''
    top_ten_metrics = compute_top_experience_metrics(user_experience_cleaned=user_experience)
    print(top_ten_metrics)


    '''Task 3.3'''


    '''Task 3.4'''