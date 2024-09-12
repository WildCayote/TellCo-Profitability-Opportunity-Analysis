import os
import pandas as pd
import streamlit as st
from utils.data import init_connection, run_query

# add info about the dashboard
st.markdown("## Welcome to the TellCo Profitability Opportunity Analysis Dashboard!")
st.markdown("This interactive app provides an in-depth analysis of user engagement, experience, and satisfaction data to help identify key growth opportunities for TellCo in the Republic of Pefkakia.")

# obtain values form environment variables
host = os.getenv("DB_HOST")
user_name = os.getenv("DB_USER")
passowrd = os.getenv("DB_PASSWORD")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

# initialize connection
conn = init_connection(host=host, port=port, user_name=user_name, password=passowrd, database_name=database)

# create tabs
tab1, tab2, tab3, tab4 = st.tabs(["User Overview Analysis", "User Engagement Analysis", "User Experience Analysis", "User Satisfaction Analysis"])

with tab1:
    # Query to get the top 10 handsets
    query_top_handsets = "SELECT \"Handset Type\", COUNT(*) AS \"Count\" FROM xdr_data GROUP BY \"Handset Type\" ORDER BY \"Count\" DESC LIMIT 10"
    top_handsets = run_query(query_top_handsets, connection=conn)

    if top_handsets is not None:
        top_handsets_df = pd.DataFrame(top_handsets, columns=['Handset Type', 'Count'])
        st.subheader("Top Handsets")
        st.bar_chart(top_handsets_df, x='Handset Type', y='Count')
    else:
        st.error("Failed to fetch top handsets data.")

    # Query to get the top 10 handset manufacturers
    query_top_manufacturers = "SELECT \"Handset Manufacturer\", COUNT(*) AS \"Count\" FROM xdr_data GROUP BY \"Handset Manufacturer\" ORDER BY \"Count\" DESC LIMIT 10"
    top_manufacturers = run_query(query_top_manufacturers, connection=conn)

    if top_manufacturers is not None:
        top_manufacturers_df = pd.DataFrame(top_manufacturers, columns=['Handset Manufacturer', 'Count'])
        st.subheader("Top Handsets by Manufacturers")
        st.bar_chart(top_manufacturers_df, x='Handset Manufacturer', y='Count')
    else:
        st.error("Failed to fetch top handset manufacturers data.")

    # Query to get users with the top number of sessions
    query_top_sessions = "SELECT \"MSISDN/Number\", COUNT(*) AS \"Session Count\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Session Count\" DESC LIMIT 10"
    top_sessions = run_query(query_top_sessions, connection=conn)

    if top_sessions is not None:
        top_sessions_df = pd.DataFrame(top_sessions, columns=['MSISDN/Number', 'Session Count'])
        st.subheader("Users with the Top Number of Sessions")
        st.table(top_sessions_df)
    else:
        st.error("Failed to fetch top sessions data.")

    # Query to get users with the top total data usage
    query_top_data = "SELECT \"MSISDN/Number\", SUM(\"Total DL (Bytes)\") + SUM(\"Total UL (Bytes)\") AS \"Total Data Used\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Total Data Used\" DESC LIMIT 10"
    top_data = run_query(query_top_data, connection=conn)

    if top_data is not None:
        top_data_df = pd.DataFrame(top_data, columns=['MSISDN/Number','Total Data Used'])
        st.subheader("Users with the Top Total Data Used")
        st.table(top_data_df)
    else:
        st.error("Failed to fetch top data usage data.")

    # Query to get users with the top total duration of sessions
    query_top_duration = "SELECT \"MSISDN/Number\", SUM(\"Dur. (ms)\") AS \"Total Duration\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Total Duration\" DESC LIMIT 10"
    top_duration = run_query(query_top_duration, connection=conn)

    if top_duration is not None:
        top_duration_df = pd.DataFrame(top_duration, columns=['MSISDN/Number', 'Total Duration'])
        st.subheader("Users with the Top Total Duration of Sessions")
        st.table(top_duration_df)
    else:
        st.error("Failed to fetch top duration data.")

    # Query to get users with the top average duration of sessions
    query_top_avg_duration = "SELECT \"MSISDN/Number\", AVG(\"Dur. (ms)\") AS \"Average Duration\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Average Duration\" DESC LIMIT 10"
    top_avg_duration = run_query(query_top_avg_duration, connection=conn)

    if top_avg_duration is not None:
        top_avg_duration_df = pd.DataFrame(top_avg_duration, columns=['MSISDN/Number', 'Average Duration'])
        st.subheader("Users with the Top Average Duration of Sessions")
        st.table(top_avg_duration_df)
    else:
        st.error("Failed to fetch top average duration data.")

with tab2:
    # List of engagement metrics for selection
    engagement_metrics = ['Number of Sessions', 'Total Duration', 'Total Data Volume']
    selected_engagement_metric = st.selectbox("Select an engagement metric:", engagement_metrics)

    # Determine the query based on the selected engagement metric
    if selected_engagement_metric == 'Number of Sessions':
        query_engagement = "SELECT \"MSISDN/Number\", COUNT(*) AS \"Session Count\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Session Count\" DESC LIMIT 10"
    elif selected_engagement_metric == 'Total Duration':
        query_engagement = "SELECT \"MSISDN/Number\", SUM(\"Dur. (ms)\") AS \"Total Duration\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Total Duration\" DESC LIMIT 10"
    elif selected_engagement_metric == 'Total Data Volume':
        query_engagement = "SELECT \"MSISDN/Number\", SUM(\"Total DL (Bytes)\") + SUM(\"Total UL (Bytes)\") AS \"Total Data Volume\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Total Data Volume\" DESC LIMIT 10"
    else:
        st.error("Invalid engagement metric selected.")

    # Execute the query and fetch the engagement data
    engagement_data = run_query(query_engagement, connection=conn)

    if engagement_data is not None:
        # Convert the fetched data into a DataFrame
        engagement_df = pd.DataFrame(engagement_data, columns=['MSISDN/Number', selected_engagement_metric])
        st.subheader(f"Top 10 Users by {selected_engagement_metric}")
        st.table(engagement_df)
    else:
        st.error("Failed to fetch engagement data.")

with tab3:
    # List of experience metrics for selection
    experience_metrics = ['Avg RTT DL', 'Avg RTT UL', 'Avg Bearer TP DL', 'Avg Bearer TP UL']
    selected_experience_metric = st.selectbox("Select an experience metric:", experience_metrics)

    # Determine the query based on the selected experience metric
    if selected_experience_metric == 'Avg RTT DL':
        query_experience = "SELECT \"MSISDN/Number\", \"Avg RTT DL (ms)\" AS \"Average RTT DL\" FROM xdr_data ORDER BY \"Average RTT DL\" DESC LIMIT 10"
    elif selected_experience_metric == 'Avg RTT UL':
        query_experience = "SELECT \"MSISDN/Number\", \"Avg RTT UL (ms)\" AS \"Average RTT UL\" FROM xdr_data ORDER BY \"Average RTT UL\" DESC LIMIT 10"
    elif selected_experience_metric == 'Avg Bearer TP DL':
        query_experience = "SELECT \"MSISDN/Number\", \"Avg Bearer TP DL (kbps)\" AS \"Average Bearer TP DL\" FROM xdr_data ORDER BY \"Average Bearer TP DL\" DESC LIMIT 10"
    elif selected_experience_metric == 'Avg Bearer TP UL':
        query_experience = "SELECT \"MSISDN/Number\", \"Avg Bearer TP UL (kbps)\" AS \"Average Bearer TP UL\" FROM xdr_data ORDER BY \"Average Bearer TP UL\" DESC LIMIT 10"
    else:
        st.error("Invalid experience metric selected.")

    # Execute the query and fetch the experience data
    experience_data = run_query(query_experience, connection=conn)

    if experience_data is not None:
        # Convert the fetched data into a DataFrame
        experience_df = pd.DataFrame(experience_data, columns=['MSISDN/Number', selected_experience_metric])
        st.subheader(f"Top 10 Users by {selected_experience_metric}")
        st.table(experience_df)
    else:
        st.error("Failed to fetch experience data.")

with tab4:
    # List of satisfaction metrics for selection
    satisfaction_metrics = ['Engagement Score', 'Experience Score']
    selected_satisfaction_metric = st.selectbox("Select a satisfaction metric:", satisfaction_metrics)

    # Determine the query based on the selected satisfaction metric
    if selected_satisfaction_metric == 'Engagement Score':
        query_satisfaction = "SELECT \"MSISDN/Number\", COUNT(*) AS \"Engagement Score\" FROM xdr_data GROUP BY \"MSISDN/Number\" ORDER BY \"Engagement Score\" DESC LIMIT 10"
    elif selected_satisfaction_metric == 'Experience Score':
        query_satisfaction = "SELECT \"MSISDN/Number\", \"Avg RTT DL (ms)\" + \"Avg RTT UL (ms)\" + \"Avg Bearer TP DL (kbps)\" + \"Avg Bearer TP UL (kbps)\" AS \"Experience Score\" FROM xdr_data ORDER BY \"Experience Score\" DESC LIMIT 10"
    else:
        st.error("Invalid satisfaction metric selected.")

    # Execute the query and fetch the satisfaction data
    satisfaction_data = run_query(query_satisfaction, connection=conn)

    if satisfaction_data is not None:
        # Convert the fetched data into a DataFrame
        satisfaction_df = pd.DataFrame(satisfaction_data, columns=['MSISDN/Number', selected_satisfaction_metric])
        st.subheader(f"Top 10 Users by {selected_satisfaction_metric}")
        st.table(satisfaction_df)
    else:
        st.error("Failed to fetch satisfaction data.")
