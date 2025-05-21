import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from typing import Dict, List, Optional, Union
from pathlib import Path
from typing import Dict, Any  # Use Any instead of Union[pd, np, ...]

def import_libraries() -> Dict[str, Any]:
    """
    Import commonly used libraries for the Solar Farm Investment Analysis project.

    Returns
    -------
    dict
        Dictionary mapping library aliases to their respective modules.

    Raises
    ------
    ImportError
        If a required library is not installed, with instructions for installation.
    """
    libraries = {}
    lib_configs = [
        ('pd', 'pandas', 'pandas==2.2.2'),
        ('np', 'numpy', 'numpy==1.26.4'),
        ('plt', 'matplotlib.pyplot', 'matplotlib==3.8.4'),
        ('sns', 'seaborn', 'seaborn==0.13.2'),
        ('stats', 'scipy.stats', 'scipy==1.13.0'),
        ('px', 'plotly.express', 'plotly==5.22.0'),
        ('st', 'streamlit', 'streamlit==1.38.0'),
        ('go', 'plotly.graph_objects', 'plotly==5.22.0'),
    ]

    for alias, module_name, install_cmd in lib_configs:
        try:
            module = __import__(module_name, fromlist=[''])
            libraries[alias] = module
        except ImportError:
            print(f"{module_name.split('.')[-1].capitalize()} not found. "
                  f"Install it using: pip install {install_cmd}")
    return libraries


def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame or None
        Loaded DataFrame if successful, None if the file path is invalid.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Unable to parse CSV at {file_path}")
        return None


def summarize_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for numeric columns in the DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Summary statistics for numeric columns.
    """
    return df.select_dtypes(include='number').describe()


def detect_outliers_and_missing(
    df: pd.DataFrame,
    key_columns: List[str],
    negative_threshold: float = 0.0,
    null_threshold: float = 0.05
) -> tuple[Dict[str, Dict[str, float]], Dict[str, float], List[str]]:
    """
    Analyze negative values and missing data in specified DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    key_columns : list of str
        List of column names to analyze.
    negative_threshold : float, optional
        Threshold for reporting negative values (default: 0.0).
    null_threshold : float, optional
        Threshold for reporting high null percentages (default: 0.05).

    Returns
    -------
    tuple
        - dict: Negative counts and percentages for each column.
        - dict: Missing value counts and percentages for each column.
        - list: Columns with null percentages exceeding the threshold.

    Raises
    ------
    ValueError
        If key_columns is empty or contains invalid column names.
    """
    if not key_columns or not all(col in df.columns for col in key_columns):
        raise ValueError("Invalid or empty key_columns provided")

    negatives = {
        col: {
            'count': (df[col] < 0).sum(),
            'percent': round((df[col] < 0).sum() / len(df) * 100, 2)
        } for col in key_columns
    }

    missing = df[key_columns].isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    high_null_cols = missing_pct[missing_pct > null_threshold * 100].index.tolist()

    print("Negative Values in Key Columns:")
    for col, stats in negatives.items():
        if stats['count'] > negative_threshold:
            print(f"{col}: {stats['count']} negative values ({stats['percent']}%)")

    print(f"\nColumns with >{null_threshold*100}% Nulls: {high_null_cols or 'None'}")
    return negatives, missing_pct.to_dict(), high_null_cols


def calculate_z_scores(df: pd.DataFrame, key_columns: List[str]) -> None:
    """
    Compute z-scores for key columns and identify outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    key_columns : list of str
        List of column names to compute z-scores for.

    Returns
    -------
    None
        Prints z-scores and outlier counts.
    """
    z_scores = stats.zscore(df[key_columns].fillna(df[key_columns].median()))
    outliers = (abs(z_scores) > 3).any(axis=1)
    print(f"\nZ-scores of Key Columns:\n{z_scores}")
    print(f"\nOutliers (|Z|>3): {outliers.sum()} rows")


def clean_data(df: pd.DataFrame, key_columns: List[str], country_name: str) -> pd.DataFrame:
    """
    Clean DataFrame by removing negative values and imputing missing values with median.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    key_columns : list of str
        List of column names to clean.
    country_name : str
        Name of the country for file naming.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame.

    Raises
    ------
    ValueError
        If key_columns is empty or contains invalid column names.
    """
    if not key_columns or not all(col in df.columns for col in key_columns):
        raise ValueError("Invalid or empty key_columns provided")

    df_clean = df.copy()
    for col in ['GHI', 'DNI', 'DHI']:
        if col in df_clean.columns:
            df_clean = df_clean[df_clean[col] >= 0]
    
    df_clean[key_columns] = df_clean[key_columns].fillna(df[key_columns].median())
    
    print(f"Cleaned DataFrame Shape: {df_clean.shape}")
    print("Remaining Negatives in Key Columns:")
    for col in key_columns:
        print(f"{col}: {(df_clean[col] < 0).sum()}")

    output_path = Path(f'../data/{country_name.lower()}_clean.csv')
    output_path.parent.mkdir(exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned data exported to {output_path}")
    return df_clean


def plot_time_series(df: pd.DataFrame, country_name: str) -> None:
    """
    Plot time series and monthly averages for solar and temperature data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'Timestamp', 'GHI', 'DNI', 'DHI', 'Tamb' columns.
    country_name : str
        Name of the country for plot titles.

    Returns
    -------
    None
        Displays plots.
    """
    df = df.copy()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    plt.figure(figsize=(12, 6))
    for col in ['GHI', 'DNI', 'DHI', 'Tamb']:
        if col in df.columns:
            plt.plot(df['Timestamp'], df[col], label=col)
    plt.title(f'{country_name.title()}: Solar and Temperature Over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    df['Month'] = df['Timestamp'].dt.month
    monthly = df.groupby('Month')[['GHI', 'DNI', 'DHI']].mean()
    monthly.plot(kind='bar', figsize=(10, 6), title=f'{country_name.title()}: Monthly Averages')
    plt.show()


def plot_cleaning_impact(df: pd.DataFrame, country_name: str) -> None:
    """
    Plot the impact of cleaning on ModA and ModB.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'Cleaning', 'ModA', 'ModB' columns.
    country_name : str
        Name of the country for plot titles.

    Returns
    -------
    None
        Displays plot.
    """
    cleaning_impact = df.groupby('Cleaning')[['ModA', 'ModB']].mean()
    cleaning_impact.plot(kind='bar', figsize=(8, 5), title=f'{country_name.title()}: ModA, ModB by Cleaning')
    plt.show()


def analyze_correlations(df: pd.DataFrame, country_name: str) -> None:
    """
    Plot correlation heatmap and scatter plots for environmental variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with relevant columns.
    country_name : str
        Name of the country for plot titles.

    Returns
    -------
    None
        Displays plots.
    """
    corr_cols = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB']
    corr_cols = [col for col in corr_cols if col in df.columns]
    if corr_cols:
        sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'{country_name.title()}: Correlation Heatmap')
        plt.show()

    for col in ['WS', 'WSgust', 'WD', 'RH']:
        if col in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df[col], df['GHI'], alpha=0.5)
            plt.title(f'{country_name.title()}: {col} vs. GHI')
            plt.xlabel(col)
            plt.ylabel('GHI')
            plt.show()


def plot_wind_rose(df: pd.DataFrame, country_name: str) -> None:
    """
    Plot a wind rose showing average wind speed by wind direction.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'WS' and 'WD' columns.
    country_name : str
        Name of the country for plot titles.

    Returns
    -------
    None
        Displays wind rose plot.
    """
    bins = np.linspace(0, 360, 17)
    wd_binned = pd.cut(df['WD'], bins, include_lowest=True, labels=bins[:-1])
    ws_mean = df.groupby(wd_binned, observed=True)['WS'].mean().reset_index()
    ws_mean['WD'] = ws_mean['WD'].astype(float)

    fig = go.Figure(data=[
        go.Barpolar(
            r=ws_mean['WS'],
            theta=ws_mean['WD'],
            marker_colorscale='Viridis',
            marker_color=ws_mean['WS'],
            name='Wind Speed (m/s)',
            text=[f'r={r:.1f}, θ={θ:.0f}°' for r, θ in zip(ws_mean['WS'], ws_mean['WD'])],
            hoverinfo='text',
            hoverlabel=dict(font=dict(size=12, color='black'))
        )
    ])
    fig.update_layout(
        title=f'{country_name.title()}: Wind Rose (Average WS by WD)',
        polar=dict(
            radialaxis=dict(visible=True, title='Wind Speed (m/s)'),
            angularaxis=dict(direction='clockwise', showticklabels=True)
        ),
        annotations=[
            dict(
                text='Wind Direction (degrees)',
                x=0.5, y=-0.1, xref='paper', yref='paper',
                showarrow=False, font=dict(size=12)
            )
        ],
        showlegend=True,
        width=800, height=800
    )
    fig.show()


def plot_distributions(df: pd.DataFrame, country_name: str) -> None:
    """
    Plot histograms for GHI and wind speed distributions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'GHI' and 'WS' columns.
    country_name : str
        Name of the country for plot titles.

    Returns
    -------
    None
        Displays histograms.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['GHI'], bins=50, color='blue')
    plt.title(f'{country_name.title()}: GHI Distribution')
    plt.xlabel('GHI (W/m²)')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    sns.histplot(df['WS'], bins=50, color='green')
    plt.title(f'{country_name.title()}: WS Distribution')
    plt.xlabel('Wind Speed (m/s)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_temperature_relations(df: pd.DataFrame, country_name: str) -> None:
    """
    Plot scatter plots of relative humidity vs. temperature and GHI.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'RH', 'Tamb', 'GHI' columns.
    country_name : str
        Name of the country for plot titles.

    Returns
    -------
    None
        Displays scatter plots.
    """
    for col in ['Tamb', 'GHI']:
        if col in df.columns and 'RH' in df.columns:
            plt.figure(figsize=(6, 4))
            plt.scatter(df['RH'], df[col], alpha=0.5)
            plt.title(f'{country_name.title()}: RH vs. {col}')
            plt.xlabel('RH (%)')
            plt.ylabel(f'{col} ({"°C" if col == "Tamb" else "W/m²"})')
            plt.show()


def plot_bubble_chart(df: pd.DataFrame, country_name: str) -> None:
    """
    Plot a bubble chart of GHI vs. temperature with bubble size by relative humidity.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame with 'GHI', 'Tamb', 'RH' columns.
    country_name : str
        Name of the country for plot titles.

    Returns
    -------
    None
        Displays bubble chart.
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(
        df['Tamb'], df['GHI'], s=df['RH'] * 20, alpha=0.5, c=df['RH'], cmap='viridis'
    )
    plt.colorbar(label='Relative Humidity (%)')
    plt.title(f'{country_name.title()}: GHI vs. Tamb (Bubble Size = RH)')
    plt.xlabel('Tamb (°C)')
    plt.ylabel('GHI (W/m²)')
    plt.show()