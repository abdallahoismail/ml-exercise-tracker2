import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from features.DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_preprocessed_data.pkl")

df.shape
df.head()
df.isnull().sum()

# uing walrus operator to drop rows where label is 'rest'
df.drop(rows := df[df["label"] == "rest"].index, inplace=True)
df.shape

# alternative way to drop rows where label is 'rest'
# rows_to_drop = df[df["label"] == "rest"].index
# rows_to_drop
# # the default is axis=0 which means we are dropping rows
# # we can also use the arg index=rows_to_drop to deop rows
# df.drop(rows_to_drop, inplace=True)

#  3rd way to drop rows where label is 'rest'
# df = df[df["label"] != "rest"]
# df.shape

acc_r = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)

gyr_r = np.sqrt(df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2)

df["acc_r"] = acc_r
df["gyr_r"] = gyr_r

df.columns.tolist()
# --------------------------------------------------------------
# Split data by exercise
# --------------------------------------------------------------
df_squat = df[df["label"] == "squat"]
df_ohp = df[df["label"] == "ohp"]
df_row = df[df["label"] == "row"]
df_bench = df[df["label"] == "bench"]
df_dead = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = df_bench

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()  # shows 5 reps
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()  # shows 5 reps
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()


# plotting individual sets
bench_set = df_bench[df_bench["set"] == df_bench["set"].unique()[0]]
squat_set = df_squat[df_squat["set"] == df_squat["set"].unique()[2]]
ohp_set = df_ohp[df_ohp["set"] == df_ohp["set"].unique()[0]]
row_set = df_row[df_row["set"] == df_row["set"].unique()[0]]
dead_set = df_dead[df_dead["set"] == df_dead["set"].unique()[0]]


bench_set["acc_x"].plot(title="Bench Press - Acc X")
bench_set["acc_z"].plot(title="Bench Press - Acc z")
bench_set["acc_r"].plot(title="Bench Press - Acc R")
# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

column = "acc_z"
samplingfreq = 1000 / 200  # 1000 ms / 200 ms
cutofffreq = 0.4
order = 3

lowpass = LowPassFilter()
# returns a dataframe with an additional column 'acc_r_lowpass'
# lowpass.low_pass_filter(bench_set, column, samplingfreq, cutofffreq, order=order)

# this particular bench set has 5 reps (heavy), we can see that the lowpass filter smoothens the signal
# unfiltered signal
bench_set[column].plot(title=f"Bench Press - {column} Unfiltered")
# filtered signal
lowpass.low_pass_filter(bench_set, column, samplingfreq, cutofffreq, order=order)[
    column + "_lowpass"
].plot(title=f"Bench Press - {column} Lowpass Filtered")

#  ******************************************************

column = "acc_z"
samplingfreq = 1000 / 200  # 1000 ms / 200 ms
cutofffreq = 0.4
order = 3

# this particular squat set has 10 reps (medium), we can see that the lowpass filter smoothens the signal
squat_set[column].plot(title=f"Squat - {column} Unfiltered")
# filtered signal
lowpass.low_pass_filter(squat_set, column, samplingfreq, cutofffreq, order=order)[
    column + "_lowpass"
].plot(title=f"Squat - {column} Lowpass Filtered")

#  ******************************************************

column = "acc_z"
samplingfreq = 1000 / 200  # 1000 ms / 200 ms
cutofffreq = 0.4
order = 3

# this particular ohp set has 5 reps (heavy), we can see that the lowpass filter smoothens the signal
ohp_set[column].plot(title=f"OHP - {column} Unfiltered")
# filtered signal
lowpass.low_pass_filter(ohp_set, column, samplingfreq, cutofffreq, order)[
    column + "_lowpass"
].plot(title=f"OHP - {column} Lowpass Filtered")

#  ******************************************************

column = "acc_x"
samplingfreq = 1000 / 200  # 1000 ms / 200 ms
cutofffreq = 1.4
order = 3

# this particular row set has 10 reps (medium), we can see that the lowpass filter smoothens the signal
row_set[column].plot(title=f"Row - {column} Unfiltered")
# filtered signal
lowpass.low_pass_filter(row_set, column, samplingfreq, cutofffreq, order=order)[
    column + "_lowpass"
].plot(title=f"Row - {column} Lowpass Filtered")

#  ******************************************************

column = "acc_r"
samplingfreq = 1000 / 200  # 1000 ms / 200 ms
cutofffreq = 0.4
order = 3

# this particular deadlift set has 5 reps (heavy), we can see that the lowpass filter smoothens the signal
dead_set[column].plot(title=f"Deadlift - {column} Unfiltered")
# filtered signal
lowpass.low_pass_filter(dead_set, column, samplingfreq, cutofffreq, order=order)[
    column + "_lowpass"
].plot(title=f"Deadlift - {column} Lowpass Filtered")

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
# create an instance of the lowpass class from the DataTransformation.py
# lowpass = LowPassFilter()

# samplingfreq = 1000 / 200  # 1000 ms / 200 ms = 5 Hz
# cutofffreq = 1.3  # 1 Hz

# # extracting the cols to be processed by the lpf
# cols = [col for col in df.columns if ("acc" in col) or ("gyr" in col)]

# for col in cols:
#     df_lowpass = lowpass.low_pass_filter(df, col, samplingfreq, cutofffreq, order=5)

# df_lowpass.shape
# df_lowpass.head()


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------
# .values converts the pandas series to a numpy array which is
# required by the argrelextrema function

column = "acc_r" + "_lowpass"
set_to_count = dead_set

peaks = argrelextrema(set_to_count[column].values, np.greater_equal, order=10)

len(peaks[0])  # number of peaks found

# peaks is a tuple of arrays, we need the first array which contains the indices of the peaks
# peaks[0] contains the indices of the peaks

troughs = argrelextrema(set_to_count[column].values, np.less_equal, order=10)
len(troughs[0])  # number of troughs found


def count_repetitions(
    data, column, samplingfreq=5, cutofffreq=0.4, order=10, plot=True
):
    """
    Count the number of repetitions in a given exercise set using local maxima detection.
    Parameters:
    - data: pd.DataFrame containing the exercise data.
    - column: str, the name of the column to analyze.
    - samplingfreq: float, the sampling frequency of the data.
    - cutofffreq: float, the cutoff frequency for the low-pass filter.
    - order: int, the number of points to compare on each side for peak detection.
    - plot: bool, whether to plot the results.
    Returns:
    - int, the number of repetitions detected.
    """

    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")

    # Instantiate filter within the function
    lowpass_filter = LowPassFilter()

    # Apply the low-pass filter
    lowpass_data = lowpass_filter.low_pass_filter(
        data, column, samplingfreq, cutofffreq, order=3
    )

    filtered_column = column + "_lowpass"

    # Find peaks in the filtered data
    peaks_indices = argrelextrema(
        lowpass_data[filtered_column].values, np.greater_equal, order=order
    )[0]

    if plot:
        # Use the ax object for all plotting
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), dpi=200)

        # Plot the filtered signal
        ax.plot(
            lowpass_data.index,
            lowpass_data[filtered_column],
            label=f"{column} Lowpass Filtered",
            color="blue",
        )

        # Correctly plot the detected peaks on the filtered signal
        ax.plot(
            lowpass_data.index[peaks_indices],
            lowpass_data[filtered_column].iloc[peaks_indices],
            "o",
            markersize=12,
            color="black",
            label="Detected Peaks",
        )

        exercise = (
            lowpass_data["label"].iloc[0].title()
            if "label" in lowpass_data.columns
            else "Exercise"
        )
        category = (
            lowpass_data["category"].iloc[0].title()
            if "category" in lowpass_data.columns
            else "Category"
        )

        ax.set_title(
            f"{exercise} - {category} - Detected Repetitions: {len(peaks_indices)}"
        )
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.legend()
        plt.show()

    return len(peaks_indices)


# testing the function
# samplingfreq = 1000 / 200
count_repetitions(dead_set, "acc_r", order=6)
count_repetitions(bench_set, "acc_z", order=10)
count_repetitions(squat_set, "acc_z", order=8)
count_repetitions(ohp_set, "acc_z", order=10)
count_repetitions(row_set, "acc_y", order=3)

count_repetitions(squat_set, "gyr_z", order=5)


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
df[df["category"] == "medium"].head()

# vectorized way to create a new column 'reps' based on the 'category' column
df["reps"] = np.where(df["category"] == "heavy", 5, 10)
df.shape

# using lambda function to create a new column 'reps' based on the 'category' column


# create a grouped df
grouping_cols = ["set", "label", "category"]
grouped_df = df.groupby(grouping_cols)["reps"].max().reset_index()

grouped_df.shape
grouped_df.head()

grouped_df[grouped_df["category"] == "heavy"]["reps"].unique()
grouped_df[grouped_df["category"] == "medium"]["reps"].unique()


def count_repetitions(data, column, samplingfreq=5, cutofffreq=0.4, order=10):
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in dataframe.")
    # Instantiate filter within the function
    lowpass_filter = LowPassFilter()

    # Apply the low-pass filter
    lowpass_data = lowpass_filter.low_pass_filter(
        data, column, samplingfreq, cutofffreq, order=3
    )

    filtered_column = column + "_lowpass"

    # Find peaks in the filtered data
    peaks = argrelextrema(
        lowpass_data[filtered_column].values, np.greater_equal, order=order
    )[0]
    return len(peaks)


# count_repetitions(bench_set, "acc_z", order=10)

# Apply the count_repetitions function to each group and store the results
results = pd.DataFrame(columns=["set", "label", "category", "predicted_reps"])
for (set_id, label, category), group in df.groupby(grouping_cols):
    if label == "bench":
        column = "acc_z"
        order = 10
    elif label == "squat":
        column = "acc_z"
        order = 8
    elif label == "ohp":
        column = "acc_z"
        order = 10
    elif label == "row":
        column = "acc_y"
        order = 3
    elif label == "dead":
        column = "acc_r"
        order = 6
    else:
        continue  # Skip unknown labels

    rep_count = count_repetitions(group, column, samplingfreq, cutofffreq, order)
    results.loc[len(results)] = [set_id, label, category, rep_count]

results.shape

results

grouped_df["predicted_reps"] = results["predicted_reps"]
grouped_df
# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
err = mean_absolute_error(grouped_df["reps"], grouped_df["predicted_reps"])
err = np.round(err, 2)
err


# create a function that takes acc an gyr data, classifies the exercise,
# counts the repetitions and returns the error
# the function uses the trained model from the train_model.py
# and the count_repetitions function from this file
def detect_exercise_and_count_reps(data):
    # from features.classify_exercise import classify_exercise
    from joblib import load

    # Load the trained model
    model = load("../../models/random_forest_model.pkl")

    # Classify the exercise
    exercise = model.predict(data.drop(columns=["label", "set", "category"]))[0]

    # Determine the column and order based on the classified exercise
    if exercise == "bench":
        column = "acc_z"
        order = 10
    elif exercise == "squat":
        column = "acc_z"
        order = 8
    elif exercise == "ohp":
        column = "acc_z"
        order = 10
    elif exercise == "row":
        column = "acc_y"
        order = 3
    elif exercise == "dead":
        column = "acc_r"
        order = 6
    else:
        raise ValueError("Unknown exercise type")

    # Count repetitions
    rep_count = count_repetitions(data, column, samplingfreq, cutofffreq, order)

    return exercise, rep_count
    
# --------------------------------------------------------------





# import numpy as np
# from scipy.signal import argrelextrema

# abc = np.array([2, 1, 2, 3, 2, 0, 1, 0])

# # Find local maxima
# maxima_indices = argrelextrema(abc, np.greater)
# print(f"Indices of local maxima: {maxima_indices}")

# # Find local minima
# minima_indices = argrelextrema(abc, np.less)
# print(f"Indices of local minima: {minima_indices}")
