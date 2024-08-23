
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.dates as mdates
def load_data(file_paths):
    dataframes = {}
    for country, path in file_paths.items():
        dataframes[country] = pd.read_csv(path)
    return dataframes

file_paths = {
    'Benin': '../data/benin-malanville.csv',
    'Sierra_Leone': '../data/sierraleone-bumbuna.csv',
    'Togo': '../data/togo-dapaong_qc.csv'
}

dfs = load_data(file_paths)


# converts the 'Timestamp' column to datetime format
def convert_timestamp(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    return df



# Check for Missing Values and Outliers
def data_check(dfs):
    for country, df in dfs.items():
        print(f"\nData Quality Check for {country}:")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"Missing Values: {missing_values.sum()}")
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        
        # Check for negative values and outliers
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Check for negative values
            negative_values = df[df[col] < 0][col]
            if len(negative_values) > 0:
                print(f"\nNegative values in {col}:")
                print(f"Number of negative values: {len(negative_values)}")
            
            # Check for outliers using IQR method
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            if len(outliers) > 0:
                print(f"\nOutliers in {col}:")
                print(f"Number of outliers: {len(outliers)}")
                print(f"Percentage of outliers: {100 * len(outliers) / len(df):.2f}%")
        
        print("\n")  

# Time Series Analysis for GHI, DNI, DHI, and Tamb
def time_series(dfs, columns):
    for country, df in dfs.items():
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for col in columns:
            ax.plot(df.index, df[col], label=col)
        
        ax.set_title(f'Time Series Analysis - {country}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Format the x-axis to display dates correctly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate and align the tick labels so they look better
        fig.autofmt_xdate()
        
        # Use tight layout to prevent clipping of tick-labels
        plt.tight_layout()
        
        plt.show()

# Time Series Analysis for Cleaning, ModA, ModB
def plot_time_series_combined(dfs, columns):
    for country, df in dfs.items():
        fig, ax = plt.subplots(figsize=(15, 8))
        
        for col in columns:
            ax.plot(df.index, df[col], label=col)
        
        ax.set_title(f'Time Series Analysis - {country}')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Value')
        ax.legend()
        
        # Format the x-axis to display dates correctly
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotate and align the tick labels so they look better
        fig.autofmt_xdate()
        
        # Use tight layout to prevent clipping of tick-labels
        plt.tight_layout()
        
        plt.show()

# correlation analysis 
def plot_correlation_heatmap(df, country):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title(f'Correlation Heatmap - {country}')
    plt.show()

# Wind Analysis 
def plot_wind_rose(df, country):
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10, 10))
    ax.scatter(df['WD'] * np.pi / 180, df['WS'])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    plt.title(f'Wind Rose - {country}')
    plt.show()

# Histogram 
def plot_histograms(df, columns, country):
    fig, axes = plt.subplots(len(columns), 1, figsize=(15, 5*len(columns)))
    for i, col in enumerate(columns):
        df[col].hist(ax=axes[i], bins=50)
        axes[i].set_title(f'{col} Distribution - {country}')
    plt.tight_layout()
    plt.show()

# Z Score Analysis 
import numpy as np
from scipy import stats

def calculate_z_scores(df):
    return df.apply(lambda x: np.abs(stats.zscore(x)))

#Bubble charts 
def plot_bubble_chart(df, country):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['GHI'], df['Tamb'], s=df['WS']*10, alpha=0.5, c=df['RH'], cmap='viridis')
    ax.set_xlabel('GHI')
    ax.set_ylabel('Tamb')
    plt.colorbar(scatter, label='RH')
    ax.set_title(f'GHI vs Tamb vs WS (size) vs RH - {country}')
    plt.show()

# Bubble charts 
def plot_bubble_chart(df, country):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['GHI'], df['Tamb'], s=df['WS']*10, alpha=0.5, c=df['RH'], cmap='viridis')
    ax.set_xlabel('GHI')
    ax.set_ylabel('Tamb')
    plt.colorbar(scatter, label='RH')
    ax.set_title(f'GHI vs Tamb vs WS (size) vs RH - {country}')
    plt.show()
   



