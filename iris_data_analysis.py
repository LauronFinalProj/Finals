import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

RESULTS_DIR = "results"


def setup_results_directory(directory: str = RESULTS_DIR) -> None:
    """Ensure that the results directory exists."""
    os.makedirs(directory, exist_ok=True)


def redirect_stdout_to_file(filename: str):
    """Redirect all stdout output to a file."""
    log_file = open(filename, "w")
    sys.stdout = log_file
    sys.stderr = log_file
    return log_file


def load_and_clean_data() -> pd.DataFrame:
    """Load the Iris dataset and perform initial cleaning."""
    data = sns.load_dataset("iris")
    print("Initial data preview:")
    print(data.head())
    print("\nMissing values per column:")
    print(data.isnull().sum())

    data_cleaned = data.dropna()
    print("\nData preview after cleaning missing values:")
    print(data_cleaned.head())

    if len(data_cleaned) < 100:
        raise ValueError("Dataset does not contain at least 100 data points.")

    return data_cleaned


def compute_descriptive_stats(data: pd.DataFrame):
    """Compute and print mean, median, std, variance, and correlation matrix."""
    features = data.columns[:-1]  # Exclude species column
    print("\nDescriptive statistics:")

    means = np.mean(data[features], axis=0)
    print("\nMean values:")
    print(means)

    medians = np.median(data[features], axis=0)
    print("\nMedian values:")
    print(medians)

    std_devs = np.std(data[features], axis=0)
    print("\nStandard deviations:")
    print(std_devs)

    variances = np.var(data[features], axis=0)
    print("\nVariances:")
    print(variances)

    corr_matrix = np.corrcoef(data[features], rowvar=False)
    print("\nCorrelation matrix:")
    print(corr_matrix)


def perform_t_test(data: pd.DataFrame):
    """Perform independent t-test between Setosa and Versicolor sepal lengths."""
    setosa = data[data["species"] == "setosa"]["sepal_length"]
    versicolor = data[data["species"] == "versicolor"]["sepal_length"]

    print(f"\nNumber of Setosa samples: {len(setosa)}")
    print(f"Number of Versicolor samples: {len(versicolor)}")

    if len(setosa) < 2 or len(versicolor) < 2:
        print("Not enough data for t-test between Setosa and Versicolor.")
        return None, None

    t_stat, p_val = stats.ttest_ind(setosa, versicolor)
    print("\nT-test results (Setosa vs Versicolor sepal length):")
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_val:.4f}")
    return t_stat, p_val


def run_linear_regression(data: pd.DataFrame):
    """Fit and print summary of linear regression: sepal_length ~ sepal_width."""
    X = sm.add_constant(data["sepal_width"])  # Add intercept
    y = data["sepal_length"]

    model = sm.OLS(y, X).fit()
    print("\nLinear regression summary (sepal_length on sepal_width):")
    print(model.summary())
    return model


def save_plots(data: pd.DataFrame, directory: str = RESULTS_DIR):
    """Generate and save pairplot and correlation heatmap."""
    print("\nGenerating and saving plots...")

    pairplot = sns.pairplot(data, hue="species")
    pairplot.fig.suptitle("Pairplot of Iris Dataset", y=1.02)
    pairplot.savefig(os.path.join(directory, "pairplot.png"))
    plt.close()

    corr = data.iloc[:, :-1].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Iris Features")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, "correlation_heatmap.png"))
    plt.close()

    print("Plots saved successfully.")


def main():
    setup_results_directory()

    # Redirect all print output to a log file in the results directory
    log_file = redirect_stdout_to_file(os.path.join(RESULTS_DIR, "output_log.txt"))

    try:
        data = load_and_clean_data()
        compute_descriptive_stats(data)
        t_stat, p_val = perform_t_test(data)
        regression_model = run_linear_regression(data)

        # Save test results to files
        with open(os.path.join(RESULTS_DIR, "t_test_results.txt"), "w") as f:
            if t_stat is not None and p_val is not None:
                f.write("T-test: Setosa vs Versicolor sepal length\n")
                f.write(f"T-statistic: {t_stat:.4f}\n")
                f.write(f"P-value: {p_val:.4f}\n")
            else:
                f.write("Not enough data to perform t-test.\n")

        with open(os.path.join(RESULTS_DIR, "regression_summary.txt"), "w") as f:
            f.write(regression_model.summary().as_text())

        save_plots(data)

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        # Reset stdout and close log file
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_file.close()
        print(f"Analysis complete. Results and logs saved in the '{RESULTS_DIR}' directory.")


if __name__ == "__main__":
    main()
