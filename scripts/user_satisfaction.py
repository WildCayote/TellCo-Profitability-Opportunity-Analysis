import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

class UserStatisfactionCalculator:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.engagement_clusters_centers = None
        self.experience_clusters_centers = None
        
        self.normalizer = None

        self.customer_grouping = self.group_customers()

    def normalize_data(self, data: pd.DataFrame):
        if self.normalizer:
            print('Using already initialized normalizer')
            normalizer = self.normalizer
        else: 
            normalizer = Normalizer().fit(X=data)
            print('Created a normalizer')

        normalized_data = normalizer.transform(X=data)
        normalized_data = pd.DataFrame(columns=data.columns, index=data.index, data=normalized_data)
        
        return normalized_data

    def compute_euclidean_distance(self, center, data):
        # Repeat center for each row of data
        center_repeated = np.tile(center, (data.shape[0], 1))

        # Compute Euclidean distances row-wise
        distances = np.linalg.norm(data.values - center_repeated, axis=1)

        return distances

    def group_customers(self):
        customer_grouping = self.data.groupby(by="MSISDN/Number")
        return customer_grouping
    
    def aggregate_experience_metrics(self):
        # aggregate the numeric values using average
        avg_stats = self.customer_grouping.agg({
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
    
    def aggregate_engagement_metrics(self):
        customer_grouping = self.group_customers()
        customer_data = customer_grouping.agg({
            "Bearer Id": "count", # count the amount of session per sim card
            "Dur. (ms)": "sum", # calculate the total duration per sim card in the data
            "Total UL (Bytes)":  "sum", #calculate the total upload per sim card
            "Total DL (Bytes)": "sum", # calculate the total download per sim card 
        })

        # finding the total trafic(sum between total download and uplaod)
        customer_data["traffic"] = customer_data["Total UL (Bytes)"] + customer_data["Total DL (Bytes)"]

        # renaming the fields
        customer_data.rename(columns={
            "Bearer Id": "session_freq",
            "Dur. (ms)": "duration",
            "Total UL (Bytes)": "upload_tot",
            "Total DL (Bytes)": "download_tot"
        }, inplace=True)

        return customer_data

    def calculate_engagement_score(self):
        # the best experience center
        best_center = self.engagement_clusters_centers[0]

        ## now let us find the distance of users from this point
        # aggrefate the data
        engagement_agg = self.aggregate_engagement_metrics()
        
        # calculate the eucledean distance
        result = self.compute_euclidean_distance(center=best_center, data=engagement_agg)

        # add it to the respective rows
        engagement_agg['engagement_score'] = result

        return engagement_agg

    def claculate_experience_score(self):
        # the worst experience center
        worst_center = self.experience_clusters_centers[0]

        ## now let us find the distance of users from this point
        # aggregate the data
        experience_agg = self.aggregate_experience_metrics()
        experience_agg = experience_agg.drop(columns=["handset_type"])

        # calculate the eucledean distance
        result = self.compute_euclidean_distance(center=worst_center, data=experience_agg)

        # add it to the respective rows
        experience_agg['experience_score'] = result

        return experience_agg
    
    def get_experience_cluster(self):
        # obtain the experience metrics for every user
        experience_agg = self.aggregate_experience_metrics()

        # drop the one categorical data
        experience_agg = experience_agg.drop(columns='handset_type')

        # normalize the data
        normalized_experience = self.normalize_data(data=experience_agg)
        
        # initialize a clustering algorithm
        clusterer = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=7)

        # cluster the data
        experience_clusters = clusterer.fit(normalized_experience)

        # save the cluster centers
        self.experience_clusters_centers = experience_clusters.cluster_centers_

        return self.experience_clusters_centers
    
    def get_engagement_cluster(self):
        # obtain the engagement metrics for every user
        engagement_agg = self.aggregate_engagement_metrics()

        # drop the one categorical data
        try:
            engagement_agg = engagement_agg.drop(columns='handset_type')
        except Exception as e:
            pass
        # normalize the data
        normalized_engagement = self.normalize_data(data=engagement_agg)
        
        # initialize a clustering algorithm
        clusterer = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=7)

        # cluster the data
        engagement_clusters = clusterer.fit(normalized_engagement)

        # save the cluster centers
        self.engagement_clusters_centers = engagement_clusters.cluster_centers_

        return self.engagement_clusters_centers
    
    def get_satifisfaction_score(self, experience_score: pd.Series, engagemet_score: pd.Series):
        return (experience_score + engagemet_score) / 2
