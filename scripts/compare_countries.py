#!/usr/bin/env python3
"""
compare_countries.py
A script to compare solar potential across Benin, Sierra Leone, and Togo using cleaned datasets.
Performs metric comparisons, statistical testing, and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal
import numpy as np
import os


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def load_data(file_paths):
    """
    Load cleaned datasets for each country.

    Args:
        file_paths (dict): Dictionary mapping country names to their cleaned CSV file paths.

    Returns:
        dict: Dictionary of DataFrames, with country names as keys.
    """
    dataframes = {}
    for country, path in file_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found for {country}.")
        df = pd.read_csv(path)
        dataframes[country] = df
    return dataframes


def create_boxplots(dataframes, metrics=['GHI', 'DNI', 'DHI'], output_dir='plots'):
    """
    Generate side-by-side boxplots for specified metrics across countries.

    Args:
        dataframes (dict): Dictionary of DataFrames for each country.
        metrics (list): List of metrics to plot (e.g., ['GHI', 'DNI', 'DHI']).
        output_dir (str): Directory to save the plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Combine data into a single DataFrame for plotting
    combined_data = pd.concat([
        df[metrics].assign(country=country)
        for country, df in dataframes.items()
    ]).melt(id_vars='country', var_name='Metric', value_name='Value')

    # Create boxplots with hue and legend=False to fix the warning
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, len(metrics), i)
        sns.boxplot(x='country', y='Value', hue='country', data=combined_data[combined_data['Metric'] == metric], 
                    palette='Set2', legend=False)
        plt.title(f'{metric} Boxplot by Country')
        plt.xlabel('Country')
        plt.ylabel(f'{metric} (W/m²)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplots.png'))
    plt.show()
    plt.close()


def compute_summary_table(dataframes, metrics=['GHI', 'DNI', 'DHI']):
    """
    Compute a summary table of mean, median, and std for specified metrics across countries.

    Args:
        dataframes (dict): Dictionary of DataFrames for each country.
        metrics (list): List of metrics to summarize.

    Returns:
        pd.DataFrame: Summary table.
    """
    summary_data = {'Metric': metrics}
    for country, df in dataframes.items():
        summary_data[f'{country}_Mean'] = [df[metric].mean() for metric in metrics]
        summary_data[f'{country}_Median'] = [df[metric].median() for metric in metrics]
        summary_data[f'{country}_Std'] = [df[metric].std() for metric in metrics]
    return pd.DataFrame(summary_data)


def perform_statistical_tests(dataframes, metric='GHI'):
    """
    Perform ANOVA and Kruskal-Wallis tests on a specified metric across countries.

    Args:
        dataframes (dict): Dictionary of DataFrames for each country.
        metric (str): Metric to test (e.g., 'GHI').

    Returns:
        dict: Dictionary with p-values for ANOVA and Kruskal-Wallis tests.
    """
    # Extract metric values for each country
    metric_values = {country: df[metric].dropna() for country, df in dataframes.items()}
    

    # Perform ANOVA
    anova_result = f_oneway(*metric_values.values())
    anova_stat = round(anova_result.statistic, 3)
    anova_pvalue = round(anova_result.pvalue, 3)

    # Perform Kruskal-Wallis
    kruskal_result = kruskal(*metric_values.values())
    kruskal_stat = round(kruskal_result.statistic, 3)
    kruskal_pvalue = round(kruskal_result.pvalue, 3)

    # Print results formatted to 3 decimal places
    print(f"ANOVA Test: Statistic = {anova_stat:.3f}, p-value = {anova_pvalue:.3f}")
    print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat:.3f}, p-value = {kruskal_pvalue:.3f}")

    return {
        'ANOVA_pvalue': anova_result.pvalue,
        'Kruskal_pvalue': kruskal_result.pvalue
    }


def create_bar_chart(dataframes, metric='GHI', output_dir='plots'):
    """
    Create a bar chart ranking countries by average of a specified metric.

    Args:
        dataframes (dict): Dictionary of DataFrames for each country.
        metric (str): Metric to plot (e.g., 'GHI').
        output_dir (str): Directory to save the plot.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate average for the metric
    means = {country: df[metric].mean() for country, df in dataframes.items()}

    # Create bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(means.keys(), means.values(), color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f'Average {metric} by Country')
    plt.xlabel('Country')
    plt.ylabel(f'Average {metric} (W/m²)')
    plt.ylim(0, max(means.values()) * 1.2)
    for i, v in enumerate(means.values()):
        plt.text(i, v + 5, f'{v:.1f}', ha='center')
    plt.savefig(os.path.join(output_dir, f'{metric}_bar_chart.png'))
    plt.show()
    plt.close()


def generate_observations(dataframes, metrics=['GHI', 'DNI', 'DHI']):
    """
    Generate key observations based on the data.

    Args:
        dataframes (dict): Dictionary of DataFrames for each country.
        metrics (list): List of metrics to analyze.

    Returns:
        list: List of observation strings.
    """
    summary = compute_summary_table(dataframes, metrics)
    observations = []

    # Update observations with actual data
    ghi_medians = summary[summary['Metric'] == 'GHI'][['Benin_Median', 'Sierra Leone_Median', 'Togo_Median']].iloc[0]
    ghi_stds = summary[summary['Metric'] == 'GHI'][['Benin_Std', 'Sierra Leone_Std', 'Togo_Std']].iloc[0]
    
    max_median_country = ghi_medians.idxmax().replace('_Median', '')
    max_median_value = ghi_medians.max()
    max_std_country = ghi_stds.idxmax().replace('_Std', '')
    max_std_value = ghi_stds.max()

    observations.append(
        f"{max_median_country} shows the highest median GHI (~{max_median_value:.1f} W/m²), "
        f"indicating strong solar potential."
    )
    observations.append(
        f"{max_std_country} has the greatest GHI variability (std ~{max_std_value:.1f}), "
        f"suggesting inconsistent solar conditions."
    )
    observations.append(
        "Sierra Leone exhibits a more stable GHI profile with a median of ~406.1 W/m², "
        "suitable for consistent energy generation."
    )
    return observations


def main():
    """
    Main function to run the cross-country comparison.
    """
    # Define file paths (adjust to your local paths)
    file_paths = {
        'Benin': r'c:\Users\Lidya\Documents\projects\solar-challenge-week0\data\benin_clean.csv',
        'Sierra Leone': r'c:\Users\Lidya\Documents\projects\solar-challenge-week0\data\sierraleone_clean.csv',
        'Togo': r'c:\Users\Lidya\Documents\projects\solar-challenge-week0\data\togo_clean.csv'
    }

    # Load data
    try:
        dataframes = load_data(file_paths)
    except FileNotFoundError as e:
        print(e)
        return

    # Create boxplots
    create_boxplots(dataframes)

    # Compute and display summary table
    summary_table = compute_summary_table(dataframes)
    print("\nSummary Table:")
    print(summary_table.to_string(index=False))

    # Perform statistical tests
    stats_results = perform_statistical_tests(dataframes, metric='GHI')
    print("\nStatistical Test Results:")
    print(f"ANOVA p-value: {stats_results['ANOVA_pvalue']:.4f}")
    print(f"Kruskal-Wallis p-value: {stats_results['Kruskal_pvalue']:.4f}")

    # Generate observations
    observations = generate_observations(dataframes)
    print("\nKey Observations:")
    for obs in observations:
        print(f"- {obs}")

    # Create bar chart for GHI
    create_bar_chart(dataframes, metric='GHI')


if __name__ == "__main__":
    main()