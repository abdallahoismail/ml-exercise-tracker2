import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from features.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from features.TemporalAbstraction import NumericalAbstraction
from features.model_persistence_guide import ProductionReadyPCA, ProductionReadyClustering


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02.1_outliers_removed_chauvenet.pkl")
df.head()

df.shape

df.isnull().sum()

df.info()


numeric_columns = df.select_dtypes(include="float").columns.tolist()


# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------

for col in numeric_columns:
    df[col + "_filled"] = df[col].interpolate()

df.info()
df.head()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------
df[df["set"] == 25]["acc_z_filled"].plot()

df[df["set"] == 25]

# we can see 5 reps
df[df["set"] == 5]["acc_z_filled"].plot()


df[(df["category"] == "medium") & (df["set"] == 90)]["acc_z_filled"].plot()

df[df["set"] == 1]["gyr_x_filled"].plot()


# we see 10 reps
df[(df["category"] == "medium") & (df["set"] == 90)]["acc_y_filled"].plot()

# Note: when filetring a df on more than one condition us df[ () & () ]


# get the time delta for each set
timedelta = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
timedelta = timedelta.seconds


# for s in df['set'].unique():
#     duration = df[df['set'] == s].index[-1] - df[df['set'] == s].index[0]
#     duration = duration.seconds
#     df.loc[(df['set'] == s), 'set_duration'] = duration


for s in df["set"].unique():
    duration = df[df["set"] == s].index[-1] - df[df["set"] == s].index[0]
    duration = duration.seconds
    df.loc[(df["set"] == s), "set_duration"] = duration

df.head()

# for set in df["set"].unique():
#     timedelta = df[df["set"] == set].index[-1] - df[df["set"] == set].index[0]
#     duration = timedelta.seconds
#     # df.loc[df['set'] == set, 'set_duration'] = duration
#     # df.loc[df['set' == set]]['set_duration'] = duration
#     df.loc[(df["set"] == set), "set_duration"] = duration


# df['set_duration'] = df.groupby('set')['timestamp'].diff().dt.total_seconds()
# df.groupby(df['set']).index


# df.groupby('categoy you wanna group by)[column you wanna agg].agg-function
df.groupby(df["category"])["set_duration"].mean()


duration_df = df.groupby(df["category"])["set_duration"].mean()

# we know that heavy sets have 5 reps: a heavy rep lasts 3 secs
heavy_set_rep_dur = duration_df.iloc[0] / 5

# we know that medium sets have 10 reps: a medium rep lasts 2.5 secs
medium_set_rep_dur = duration_df.iloc[1] / 10

# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------
df_lowpass = df.copy()
df_lowpass.columns.tolist()

# create an instance of the lowpass class from the DataTransformation.py
lowpass = LowPassFilter()

samplingfreq = 1000 / 200  # 1000 ms / 200 ms = 5 Hz
cutofffreq = 1.3  # 1 Hz


# df_lowpass = lowpass.low_pass_filter(
#     df_lowpass, "acc_y_filled", samplingfreq, cutofffreq, order=5
# )


# plot the original and lowpass filtered data on the same graph of one of the sets

# the lowpass filter smoothes out jagged edges in the data (removes high freqs)

set_to_plot = 4
subset = df_lowpass[df_lowpass["set"] == set_to_plot]

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y_filled"], label="Original Data")
ax[0].set_title("Original Data")
ax[0].set_ylabel("Acceleration (m/s^2)")
ax[0].legend()

ax[1].plot(
    subset["acc_y_filled_lowpass"], label="Lowpass Filtered Data", color="orange"
)
ax[1].set_title("Lowpass Filtered Data")
ax[1].set_xlabel("Index")
ax[1].set_ylabel("Acceleration (m/s^2)")
ax[1].legend()
plt.show()


# extracting the cols to be processed by the lpf
# method 1
[col for col in df_lowpass.columns.tolist() if "filled" in col]

# method 2
df_lowpass.columns[df_lowpass.columns.str.contains("filled")].tolist()

# method 3
list(filter(lambda x: "filled" in x, df_lowpass.columns))

# for col in df_lowpass.columns:
#     print(col)

filled_cols = [col for col in df_lowpass.columns.tolist() if "filled" in col]

for col in filled_cols:
    df_lowpass = lowpass.low_pass_filter(
        df_lowpass, col, samplingfreq, cutofffreq, order=5
    )

df_lowpass.head()

# method 1
lowpass_cols = [col for col in df_lowpass.columns.tolist() if "lowpass" in col]

# method 2
df_lowpass.columns[df_lowpass.columns.str.contains("lowpass")].tolist()

# method 3: filter(function, iterable)
list(filter(lambda x: "lowpass" in x, df_lowpass.columns))

list(filter(lambda x: "lowpass" in x, df_lowpass.columns))

# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------
df_pca = df_lowpass.copy()

pca = ProductionReadyPCA()

# pc_value = pca.determine_pc_explained_variance(df_pca, lowpass_cols)

# plotting the explained variance - scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pc_value) + 1), pc_value, marker="o")
plt.title("Scree Plot of PCA")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance")
plt.xticks(range(1, len(pc_value) + 1))
plt.grid()
plt.show()


df_pca = pca.fit_and_apply_pca(df_pca, lowpass_cols, 3)

# save pca model to transform new data later
pca.save_pca_model("../../models/pca_model.pkl")

subset = df_pca[df_pca["set"] == 25]
subset[["pca_1", "pca_2", "pca_3"]].plot(subplots=True)
# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
df_squared = df_pca.copy()

# r_{magnitude} = sqrt{x^2 + y^2 + z^2}

acc_r = np.sqrt(
    df_squared["acc_x_filled_lowpass"] ** 2
    + df_squared["acc_y_filled_lowpass"] ** 2
    + df_squared["acc_z_filled_lowpass"] ** 2
)

gyr_r = np.sqrt(
    df_squared["gyr_x_filled_lowpass"] ** 2
    + df_squared["gyr_y_filled_lowpass"] ** 2
    + df_squared["gyr_z_filled_lowpass"] ** 2
)

df_squared["acc_r"] = acc_r
df_squared["gyr_r"] = gyr_r

subset = df_squared[df_squared["set"] == 25]

subset[["acc_r", "gyr_r"]].plot(subplots=True)


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------

df_temporal = df_squared.copy()
df_temporal.columns.tolist()

df_temporal["label"].unique().tolist()

temporal_cols = [
    col
    for col in df_temporal.columns
    if "lowpass" in col or "pca" in col or "_r" in col
]


window_size = int(
    1000 / 200
)  # the window size is 1 seconds, remember that the sampling frequency is 5 Hz (5 measurements per second)

agg_func1 = "mean"
agg_func2 = "std"


num_abstractor = NumericalAbstraction()

# df_temporal = num_abstractor.abstract_numerical(df_temporal, temporal_cols, window_size, agg_func)

# df_temporal = num_abstractor.abstract_numerical(df_temporal, temporal_cols, window_size, 'std')

# df_temporal.columns.tolist()[15:]

# df_temporal.info()

# IMPORTANT: doing moving averages like this is going to add a lot of noise in the data because the different sets & exercises will be mixed together: the first observation of bench press, for e.g., will look at the previous 4 rows that will be of a deadlift and then compute the average of these 5 records and assign it to the bench press in the MA col, but the values that went into this avg are for 2 diff exercises or sets, so we have to be careful of how to approach this. data leakage


# ~~~~~ loop - slow implementation ~~~~~
df_temporal_list = []
for set_num in df_temporal["set"].unique():
    # print(set)
    # create a copy of each unique set
    subset = df_temporal[df_temporal["set"] == set_num].copy()

    subset = num_abstractor.abstract_numerical(
        subset, temporal_cols, window_size, agg_func1
    )

    subset = num_abstractor.abstract_numerical(
        subset, temporal_cols, window_size, agg_func2
    )
    df_temporal_list.append(subset)
# concatenate the list of dataframes into one dataframe
df_temporal = pd.concat(df_temporal_list, ignore_index=True)

# ~~~~~ end of loop implementation ~~~~~

df_temporal.isnull().sum()[26:]
# df_temporal.info()
# df_temporal.columns.tolist()

# some Viz
df_temporal[df_temporal["set"] == 9][
    ["acc_y_filled_lowpass", "acc_y_filled_lowpass_temp_mean_ws_5"]
].plot()


[col for col in df_temporal.columns if "mean" in col]


# ~~~~~ vectorized - fast implementation ~~~~~
# The columns that uniquely identify each continuous data recording session

grouping_cols = ["participant", "label", "set"]

temporal_cols = [
    col
    for col in df_temporal.columns
    if ("lowpass" in col) or ("pca" in col) or (col.endswith("_r"))
]

for col in temporal_cols:
    for agg_func in ["mean", "std"]:
        new_col_name = f"{col}_rolling_{agg_func}_ws{window_size}"
        df_temporal[new_col_name] = (
            df_temporal.groupby(grouping_cols)[col]
            .rolling(window=window_size, min_periods=window_size)
            .agg(agg_func)
            .reset_index(level=grouping_cols, drop=True)
        )


#  Summary: Which to Choose?
# Choose min_periods=1 if your priority is to avoid any missing data and you can accept that the initial feature values are less stable.
# Choose min_periods=w if your priority is feature quality and consistency, and you can afford to lose the first few data points of each series or have a strategy to fill them. This is often the more statistically sound approach.

grouping_cols = ["participant", "label", "set"]

# this gives us more stable features, but we lose the first 4 rows of each set
for col in temporal_cols:
    df_temporal[f"{col}_rolling_mean_ws{window_size}"] = df_temporal.groupby(
        grouping_cols
    )[col].transform(
        lambda x: x.rolling(window=window_size, min_periods=window_size).mean()
    )
    df_temporal[f"{col}_rolling_std_ws{window_size}"] = df_temporal.groupby(
        grouping_cols
    )[col].transform(
        lambda x: x.rolling(window=window_size, min_periods=window_size).std()
    )


# ~~~~~ end of vectorized implementation ~~~~~

df_temporal.isnull().sum()
df_temporal.info()

df_temporal.shape

df_temporal.iloc[:, 30:]

df_temporal.head()


df_temporal[df_temporal["set"] == 5][
    [
        "acc_y_filled_lowpass",
        "acc_y_filled_lowpass_rolling_mean_ws5",
        "acc_y_filled_lowpass_rolling_std_ws5",
    ]
].plot()
# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

from FrequencyAbstraction import FourierTransformation

df_freq = df_temporal.copy().reset_index()

ft = FourierTransformation()

sampling_freq = 5  # 5 Hz
window_size = int(2800 / 200)  # avg length of a rep is 2.8 secs => 2.8 * 5 = 14
# 2.8 * 1000 / 200 = 14

overlap = int(window_size / 2)  # 50% overlap

freq_cols = [
    col
    for col in df_freq.columns
    if ("lowpass" in col) or ("pca" in col) or (col.endswith("_r"))
]

# Apply FFT-based feature extraction with overlapping windows
df_freq_list = []

for s in df_freq["set"].unique():
    print(f"Applying Fourier Transformation on Features for set {s}")
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = ft.abstract_frequency(subset, freq_cols, window_size, sampling_freq)
    df_freq_list.append(subset)

df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# print('Applying Fourier Transformation on Features for all sets')
# # Group once and apply transformation
# df_freq_fft = (df_freq.groupby('set', group_keys=False)
#            .apply(lambda x: ft.abstract_frequency(
#                x.reset_index(drop=True),
#                freq_cols,
#                window_size,
#                sampling_freq
#            )))

# df_freq_fft = df_freq.reset_index(drop=True).set_index('epoch (ms)', drop=True)


df_freq.isnull().sum()[30:]
df_freq.info()
df_freq.shape

df_freq = df_freq.dropna()
df_freq.shape

# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------

# to treat overlapping windows, we can skip rows
# that are adjacent and keep every second row
df_freq_non_overlap = df_freq.iloc[::2, :]

df_freq_non_overlap.shape

df_freq_non_overlap.columns.tolist()

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
from sklearn.cluster import KMeans

df_cluster = df_freq_non_overlap.copy()

cluster_cols = ["acc_x_filled_lowpass", "acc_y_filled_lowpass", "acc_z_filled_lowpass"]
max_clus_num = 10
wcss = []

for i in range(2, max_clus_num + 1):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42).fit(
        df_cluster[cluster_cols]
    )
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(range(1, max_clus_num), wcss)
plt.title("Elbow Method for Optimal Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# number of c;uster is 4

kmeans = KMeans(n_clusters=4, init="k-means++", random_state=42)

df_cluster["cluster"] = kmeans.fit_predict(df_cluster[cluster_cols])


# Plot clusters
fig = plt.figure(figsize=(15, 15), dpi=800)
ax = fig.add_subplot(projection="3d")

for c in df_cluster["cluster"].unique():
    mask = df_cluster["cluster"] == c
    ax.scatter(
        df_cluster.loc[mask, "acc_x_filled_lowpass"],
        df_cluster.loc[mask, "acc_y_filled_lowpass"],
        df_cluster.loc[mask, "acc_z_filled_lowpass"],
        label=c,
    )

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()


# fig = plt.figure(figsize=(15, 15))
# ax = fig.add_subplot(projection="3d")

# # Vectorized approach - plot all clusters at once
# for l in df_cluster['label'].unique():
#     mask = df_cluster['label'] == l
#     ax.scatter(df_cluster.loc[mask, 'acc_x_filled_lowpass'],
#               df_cluster.loc[mask, 'acc_y_filled_lowpass'],
#               df_cluster.loc[mask, 'acc_z_filled_lowpass'],
#               label=c)

# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.set_zlabel("Z-axis")
# plt.legend()
# plt.show()


fig = plt.figure(figsize=(15, 15), dpi=800)
ax = fig.add_subplot(projection="3d")
for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(
        subset["acc_x_filled_lowpass"],
        subset["acc_y_filled_lowpass"],
        subset["acc_z_filled_lowpass"],
        label=l,
    )

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03_features_extracted.pkl")


# fakedata = {
#     "age": [25, 30, 22, 45],
#     "city": ["New York", "Los Angeles", "Chicago", "Houston"],
#     "score": [88, 92, 95, 78],
# }
# index_labels = ["amy", "brad", "cathy", "david"]

# fakedata = pd.DataFrame(fakedata, index=index_labels)

# print(fakedata)

# # remember the phone book analogy
# fakedata.loc["amy"]
# fakedata.loc["amy", "city"]

# fakedata.iloc[0]

# fakedata.loc[:, "city"]
