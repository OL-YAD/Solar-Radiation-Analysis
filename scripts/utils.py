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

# replace negative values in GHI', 'DNI', 'DHI' with zero
def handle_negative_values(df, country_name, columns_to_check=['GHI', 'DNI', 'DHI']):
    print(f"\nHandling negative values for {country_name}:")
    
    for col in columns_to_check:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                print(f"  Found {negative_count} negative values in {col}")
                
                # Replace negative values with 0
                df[col] = df[col].clip(lower=0)
                
                print(f"  Replaced negative values with 0 in {col}")
            else:
                print(f"  No negative values found in {col}")
        else:
            print(f"  Column {col} not found in the dataset")
    
    return df

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
import matplotlib.pyplot as plt

def analyze_and_plot_outliers(df, country, z_threshold=3):
    def calculate_z_scores(data):
        return stats.zscore(data)
    
    z_scores = df.apply(calculate_z_scores)
    outliers = (np.abs(z_scores) > z_threshold)
    
    print(f"\nOutliers for {country} (|z-score| > {z_threshold}):")
    print(outliers.sum())
    
    # Filter columns with outliers
    columns_with_outliers = outliers.columns[outliers.sum() > 0]
    
    if len(columns_with_outliers) > 0:
        # Plotting
        n_cols = len(columns_with_outliers)
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 5*n_cols), sharex=True)
        fig.suptitle(f'Outlier Analysis for {country}', fontsize=16)
        
        axes = [axes] if n_cols == 1 else axes  # Ensure axes is always a list
        
        for i, column in enumerate(columns_with_outliers):
            ax = axes[i]
            ax.scatter(df.index, df[column], c='blue', alpha=0.5, label='Normal')
            ax.scatter(df.index[outliers[column]], df[column][outliers[column]], 
                       c='red', label='Outlier')
            ax.set_ylabel(column)
            ax.legend()
        
        plt.xlabel('Index')
        plt.tight_layout()
        plt.show()
    else:
        print("No columns with outliers to plot.")

# Bubble charts 
def plot_bubble_chart(df, country):
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df['GHI'], df['Tamb'], s=df['WS']*10, alpha=0.5, c=df['RH'], cmap='viridis')
    ax.set_xlabel('GHI')
    ax.set_ylabel('Tamb')
    plt.colorbar(scatter, label='RH')
    ax.set_title(f'GHI vs Tamb vs WS (size) vs RH - {country}')
    plt.show()


# Temperature Analysis
def analyze_humidity_impact(dfs):
    for country, df in dfs.items():
        print(f"\nAnalyzing humidity impact for {country}")
        
        # 1. Correlation analysis
        corr_matrix = df[['RH', 'Tamb', 'GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Heatmap - {country}')
        plt.tight_layout()
        plt.show()
        
        # 2. Scatter plots
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        sns.scatterplot(data=df, x='RH', y='Tamb', ax=axes[0, 0])
        axes[0, 0].set_title('RH vs Ambient Temperature')
        
        sns.scatterplot(data=df, x='RH', y='GHI', ax=axes[0, 1])
        axes[0, 1].set_title('RH vs Global Horizontal Irradiance')
        
        sns.scatterplot(data=df, x='RH', y='TModA', ax=axes[1, 0])
        axes[1, 0].set_title('RH vs Temperature of Module A')
        
        sns.scatterplot(data=df, x='RH', y='TModB', ax=axes[1, 1])
        axes[1, 1].set_title('RH vs Temperature of Module B')
        
        plt.tight_layout()
        plt.show()
        
        # 3. Time series analysis
        fig, axes = plt.subplots(3, 1, figsize=(15, 20), sharex=True)
        
        axes[0].plot(df.index, df['RH'], label='RH')
        axes[0].set_ylabel('RH (%)')
        axes[0].legend()
        
        axes[1].plot(df.index, df['Tamb'], label='Tamb')
        axes[1].plot(df.index, df['TModA'], label='TModA')
        axes[1].plot(df.index, df['TModB'], label='TModB')
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].legend()
        
        axes[2].plot(df.index, df['GHI'], label='GHI')
        axes[2].plot(df.index, df['DNI'], label='DNI')
        axes[2].plot(df.index, df['DHI'], label='DHI')
        axes[2].set_ylabel('Irradiance (W/m²)')
        axes[2].legend()
        
        plt.xlabel('Date')
        plt.title(f'Time Series of RH, Temperature, and Solar Radiation - {country}')
        plt.tight_layout()
        plt.show()
        
        # 4. Binned analysis
        df['RH_bins'] = pd.cut(df['RH'], bins=10)
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        
        df.groupby('RH_bins')['Tamb'].mean().plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Average Tamb by RH Bins')
        axes[0, 0].set_xlabel('RH Bins')
        axes[0, 0].set_ylabel('Average Tamb (°C)')
        
        df.groupby('RH_bins')['GHI'].mean().plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average GHI by RH Bins')
        axes[0, 1].set_xlabel('RH Bins')
        axes[0, 1].set_ylabel('Average GHI (W/m²)')
        
        df.groupby('RH_bins')['TModA'].mean().plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Average TModA by RH Bins')
        axes[1, 0].set_xlabel('RH Bins')
        axes[1, 0].set_ylabel('Average TModA (°C)')
        
        df.groupby('RH_bins')['TModB'].mean().plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Average TModB by RH Bins')
        axes[1, 1].set_xlabel('RH Bins')
        axes[1, 1].set_ylabel('Average TModB (°C)')
        
        plt.tight_layout()
        plt.show()
   

# drop 'Comments'
def clean_data(dfs):
    cleaned_dfs = {}
    for country, df in dfs.items():
        # Remove the 'Comments' column
        if 'Comments' in df.columns:
            df = df.drop('Comments', axis=1)
            print(f"Removed 'Comments' column from {country} dataset.")
        else:
            print(f"'Comments' column not found in {country} dataset.")
        
        # Check for any remaining missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\nRemaining missing values in {country} dataset:")
            print(missing_values[missing_values > 0])
        else:
            print(f"\nNo missing values remain in {country} dataset.")
        
        # Store the cleaned dataframe
        cleaned_dfs[country] = df
    
    return cleaned_dfs
