import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import scipy
import scipy.special
from sklearn.neighbors import LocalOutlierFactor
import warnings

warnings.filterwarnings("ignore")

df = pd.read_pickle("../../data/interim/01_preprocessed_data.pkl")
df.head()

# set rcParams for better visualization
plt.rcParams["figure.figsize"] = (20, 12)
plt.rcParams["axes.grid"] = True
plt.style.use("fivethirtyeight")
plt.rcParams["figure.dpi"] = 200

df[["acc_x", "label"]].boxplot(by="label", figsize=(10, 6))
plt.show()

# box plot for acc_x by label using plt
plt.figure(figsize=(10, 6))
plt.boxplot(
    [df[df["label"] == label]["acc_x"] for label in df["label"].unique()],
    labels=df["label"].unique(),
)
plt.title("Boxplot of acc_x by label")
plt.xlabel("Label")
plt.ylabel("acc_x")
plt.show()


####################################################
numeric_cols = df.select_dtypes(include=["float"]).columns.tolist()

numeric_cols

df[numeric_cols]

df[numeric_cols[:3] + ["label"]].boxplot(by="label", figsize=(20, 12), layout=(1, 3))
plt.show()


#########################################################


def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


col = "acc_x"
dataset = mark_outliers_iqr(df, col)
dataset.head()

plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)


for col in numeric_cols:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)


# the chauvenet method assumes that the numeric features are normally distributed
# let's visualize the numeric features to see if this assumption holds
for col in numeric_cols:
    plt.hist(df[col], bins=50, alpha=0.5, label=col)
    plt.title(f"Histogram of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()


# this method assumes Gaussian distribution of the numeric features
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.

    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []
    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset


# we see that this method yield better outlier detection results
for col in numeric_cols:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)


# detecting outliers using Local Outlier Factor (LOF)
# more robust to noise, especially in high-dimensional datasets
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlier
def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores


dataset, outliers, X_scores = mark_outliers_lof(df, numeric_cols)

dataset.head()

for col in numeric_cols:
    plot_binary_outliers(dataset, col, "outlier_lof", reset_index=True)


# looking at the outliers in a numeric feature for a specific label (exercise)

# debugged by me
exercise = "bench"
for col in numeric_cols:
    dataset, outliers, X_scores = mark_outliers_lof(df[df["label"] == exercise], [col])
    plot_binary_outliers(dataset, col, "outlier_lof", reset_index=True)


# debugged by Gemini
exercise = "bench"

# 1. Filter the DataFrame for the specific exercise
exercise_df = df[df["label"] == exercise].copy()

# 2. Loop through each numeric column to find and plot outliers
for col in numeric_cols:
    # 3. Call LOF for a SINGLE column by passing it as a list `[col]`
    # and correctly unpack the 3 return values.
    # We only need the dataframe, so we use _ for the other values.
    temp_df, _, _ = mark_outliers_lof(exercise_df, [col])

    # 4. Plot the results using the correct outlier column name: "outlier_lof"
    plot_binary_outliers(temp_df, col, "outlier_lof", reset_index=True)


## now it is my decision to choose which outlier detection method to use
# for now I will use the chauvenet method for the numeric features
# later I will test which method works best for the dataset and gives the highest accuracy  for the model

# --------------------------------------------------------------
# treat outliers in each numeric column for each label
# --------------------------------------------------------------

col = "gyr_z"
dataset = mark_outliers_chauvenet(
    df, col
)  # returns a dataframe with an extra column 'gyr_z_outlier'
dataset.head()

# boolean indexing to filter out the outliers
# dataset[col + '_outlier'] returns the bool col
# dataset[dataset[col + '_outlier']] returns dataset where each row is an outlier: bool col = True
dataset[dataset[col + "_outlier"]]
len(dataset[dataset[col + "_outlier"]])  # number of outliers

dataset.loc[dataset[col + "_outlier"], col] = np.nan  # set the outliers to NaN

# works
# dataset.loc[dataset[col + "_outlier"], :]

# returns value error
# dataset.loc[dataset[dataset[col + '_outlier']], :]

dataset[dataset[col + "_outlier"]]
dataset.info()

outliers_removed_df = df.copy()

for col in numeric_cols:
    for exer in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == exer], col)
        # returns a dataframe with an extra column 'gyr_z_outlier'
        dataset.loc[dataset[col + "_outlier"], col] = np.nan
        # set the outliers to NaN
        outliers_removed_df.loc[(outliers_removed_df["label"] == exer), col] = dataset[
            col
        ]

        n_outliers = len(dataset) - len(dataset.dropna())

        print(f"Removed {n_outliers} outliers from {col} for {exer}")


# vectorized implementation of the above code
# Create a copy to store the results
outliers_removed_df = df.copy()

# Loop through each numeric column
for col in numeric_cols:
    # --- Vectorized Outlier Calculation ---
    # This part remains the same; it's fast and efficient.
    mean = df.groupby("label")[col].transform("mean")
    std = df.groupby("label")[col].transform("std")
    n = df.groupby("label")[col].transform("count")
    criterion = 1.0 / (2 * n)
    deviation = abs(df[col] - mean) / std
    high = deviation / (2**0.5)
    low = -high
    prob = 1.0 - 0.5 * (scipy.special.erf(high) - scipy.special.erf(low))
    outlier_mask = prob < criterion

    # Apply the mask to the entire column at once
    outliers_removed_df.loc[outlier_mask, col] = np.nan

    # --- Per-Label Reporting ---
    # Now, group the boolean mask by label and sum the results to get the count for each exercise.
    print(f"Processing column: '{col}'")

    # Group the boolean mask by the 'label' and sum the `True` values
    per_exercise_counts = outlier_mask.groupby(df["label"]).sum()

    # Iterate through the results and print the breakdown
    for exercise, count in per_exercise_counts.items():
        if count > 0:
            print(f"  - Removed {count} outliers from '{exercise}'")

    print("-" * 40)  # Separator for readability


outliers_removed_df.info()

# export the processed dataframe
outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")
