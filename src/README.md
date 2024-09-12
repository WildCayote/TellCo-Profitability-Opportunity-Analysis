# TellCo Profitability Opportunity Analysis App

## Overview

The TellCo Profitability Opportunity Analysis App is designed to provide insights into user behavior and network performance, helping to identify opportunities for improving profitability. The app leverages data analysis to present key metrics and trends in an interactive and user-friendly interface.

## Key Features

- **User Overview Analysis**: Visualize the top handsets, handset manufacturers, and users with the highest number of sessions, data usage, and session durations.
- **Engagement Metrics**: Select and analyze different engagement metrics such as the number of sessions, total duration, and total data volume.
- **User Experience Analysis**: Evaluate user experience metrics including average round-trip time (RTT) for downloads and uploads, and average bearer throughput for downloads and uploads.
- **User Satisfaction Analysis**: Assess user satisfaction through engagement and experience scores, identifying top users based on these metrics.

## How to Use

1. **User Overview Analysis**:

   - Navigate to the "User Overview Analysis" tab.
   - View the top handsets, handset manufacturers, and users with the highest number of sessions, data usage, and session durations.
   - Interactive charts and tables provide a clear visualization of the data.

2. **Engagement Metrics**:

   - Go to the "Engagement Metrics" tab.
   - Select an engagement metric from the dropdown menu.
   - The app will display the top 10 users based on the selected metric in a table format.

3. **User Experience Analysis**:

   - Open the "User Experience Analysis" tab.
   - Choose an experience metric from the dropdown menu.
   - The app will show the top 10 users based on the selected experience metric in a table format.

4. **User Satisfaction Analysis**:
   - Access the "User Satisfaction Analysis" tab.
   - Select a satisfaction metric from the dropdown menu.
   - The app will present the top 10 users based on the selected satisfaction metric in a table format.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/TellCo-Profitability-Opportunity-Analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd TellCo-Profitability-Opportunity-Analysis
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the App

1. Ensure your database connection details are correctly configured.
2. Run the Streamlit app:
   ```bash
   streamlit run /app/main.py
   ```
