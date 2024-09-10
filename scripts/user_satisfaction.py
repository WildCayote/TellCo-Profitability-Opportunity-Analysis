import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer

class UserStatisfactionCalculator:
    """
    A class to calculate user satisfaction scores based on engagement and experience metrics. 
    This class supports data normalization, customer grouping, metric aggregation, clustering, 
    and satisfaction score calculation.

    Attributes:
        data (pd.DataFrame): The dataset containing user engagement and experience data.
        engagement_clusters_centers (np.ndarray): The centers of the engagement clusters.
        experience_clusters_centers (np.ndarray): The centers of the experience clusters.
        normalizer (Normalizer): The normalizer used for data normalization.
        customer_grouping (pd.core.groupby.DataFrameGroupBy): Grouping of customers based on their MSISDN/Number.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes the UserStatisfactionCalculator with the input data.

        Args:
            data (pd.DataFrame): A DataFrame containing user data.
        """
        self.data = data
        self.engagement_clusters_centers = None
        self.experience_clusters_centers = None
        self.normalizer = None
        self.customer_grouping = self.group_customers()

    def normalize_data(self, data: pd.DataFrame):
        """
        Normalizes the input data using the Normalizer from scikit-learn.

        Args:
            data (pd.DataFrame): The DataFrame to normalize.

        Returns:
            pd.DataFrame: The normalized data.
        """
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
        """
        Computes the Euclidean distance between the center point and all rows in the data.

        Args:
            center (np.ndarray): The center point.
            data (pd.DataFrame): The data for which distances are calculated.

        Returns:
            np.ndarray: The calculated distances for each row in the data.
        """
        center_repeated = np.tile(center, (data.shape[0], 1))
        distances = np.linalg.norm(data.values - center_repeated, axis=1)
        return distances

    def group_customers(self):
        """
        Groups the data by the 'MSISDN/Number' field.

        Returns:
            pd.core.groupby.DataFrameGroupBy: The grouped data.
        """
        customer_grouping = self.data.groupby(by="MSISDN/Number")
        return customer_grouping

    def aggregate_experience_metrics(self):
        """
        Aggregates experience metrics by calculating the mean for each customer.

        Returns:
            pd.DataFrame: The aggregated experience metrics.
        """
        avg_stats = self.customer_grouping.agg({
            "Avg RTT DL (ms)": "mean",
            "Avg RTT UL (ms)": "mean",
            "TCP DL Retrans. Vol (Bytes)": "mean",
            "TCP UL Retrans. Vol (Bytes)": "mean",
            "Avg Bearer TP DL (kbps)": "mean",
            "Avg Bearer TP UL (kbps)": "mean",
            "Handset Type": lambda x: x.mode().iloc[0] # most frequent handset type
        })

        avg_stats["avg_rtt"] = avg_stats[["Avg RTT DL (ms)", "Avg RTT UL (ms)"]].mean(axis=1)
        avg_stats["avg_tcp_rt"] = avg_stats[["TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)"]].mean(axis=1)
        avg_stats["avg_throughput"] = avg_stats[["Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"]].mean(axis=1)
        avg_stats = avg_stats.rename(columns={"Handset Type": "handset_type"})
        avg_stats = avg_stats.drop(columns=["Avg RTT DL (ms)", "Avg RTT UL (ms)", "TCP DL Retrans. Vol (Bytes)", "TCP UL Retrans. Vol (Bytes)", "Avg Bearer TP DL (kbps)", "Avg Bearer TP UL (kbps)"])
        
        return avg_stats

    def aggregate_engagement_metrics(self):
        """
        Aggregates engagement metrics by calculating session frequency, total duration, and total traffic.

        Returns:
            pd.DataFrame: The aggregated engagement metrics.
        """
        customer_grouping = self.group_customers()
        customer_data = customer_grouping.agg({
            "Bearer Id": "count",
            "Dur. (ms)": "sum",
            "Total UL (Bytes)": "sum",
            "Total DL (Bytes)": "sum",
        })

        customer_data["traffic"] = customer_data["Total UL (Bytes)"] + customer_data["Total DL (Bytes)"]
        customer_data.rename(columns={
            "Bearer Id": "session_freq",
            "Dur. (ms)": "duration",
            "Total UL (Bytes)": "upload_tot",
            "Total DL (Bytes)": "download_tot"
        }, inplace=True)

        return customer_data

    def calculate_engagement_score(self):
        """
        Calculates the engagement score for each customer based on Euclidean distance from the engagement cluster center.

        Returns:
            pd.DataFrame: The DataFrame containing engagement scores.
        """
        best_center = self.engagement_clusters_centers[0]
        engagement_agg = self.aggregate_engagement_metrics()
        result = self.compute_euclidean_distance(center=best_center, data=engagement_agg)
        engagement_agg['engagement_score'] = result
        return engagement_agg

    def claculate_experience_score(self):
        """
        Calculates the experience score for each customer based on Euclidean distance from the experience cluster center.

        Returns:
            pd.DataFrame: The DataFrame containing experience scores.
        """
        worst_center = self.experience_clusters_centers[0]
        experience_agg = self.aggregate_experience_metrics()
        experience_agg = experience_agg.drop(columns=["handset_type"])
        result = self.compute_euclidean_distance(center=worst_center, data=experience_agg)
        experience_agg['experience_score'] = result
        return experience_agg

    def get_experience_cluster(self):
        """
        Clusters the customers based on their experience metrics using KMeans clustering.

        Returns:
            np.ndarray: The cluster centers for the experience metrics.
        """
        experience_agg = self.aggregate_experience_metrics()
        experience_agg = experience_agg.drop(columns='handset_type')
        normalized_experience = self.normalize_data(data=experience_agg)
        clusterer = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=7)
        experience_clusters = clusterer.fit(normalized_experience)
        self.experience_clusters_centers = experience_clusters.cluster_centers_
        return self.experience_clusters_centers

    def get_engagement_cluster(self):
        """
        Clusters the customers based on their engagement metrics using KMeans clustering.

        Returns:
            np.ndarray: The cluster centers for the engagement metrics.
        """
        engagement_agg = self.aggregate_engagement_metrics()
        try:
            engagement_agg = engagement_agg.drop(columns='handset_type')
        except Exception as e:
            pass
        normalized_engagement = self.normalize_data(data=engagement_agg)
        clusterer = KMeans(n_clusters=3, init='k-means++', n_init=20, random_state=7)
        engagement_clusters = clusterer.fit(normalized_engagement)
        self.engagement_clusters_centers = engagement_clusters.cluster_centers_
        return self.engagement_clusters_centers

    def get_satifisfaction_score(self, experience_score: pd.Series, engagemet_score: pd.Series):
        """
        Calculates the satisfaction score for each customer by averaging their experience and engagement scores.

        Args:
            experience_score (pd.Series): The experience score for each customer.
            engagemet_score (pd.Series): The engagement score for each customer.

        Returns:
            pd.Series: The calculated satisfaction score for each customer.
        """
        return (experience_score + engagemet_score) / 2
