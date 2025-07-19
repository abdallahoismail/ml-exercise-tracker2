import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------
# the nake_dataset.py script is located in src/data/ so we need to go up (back) two directories to access the raw files
single_file_acc_path = "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
single_file_acc = pd.read_csv(single_file_acc_path)
single_file_acc.head()

single_file_gyr_path = "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
single_file_gyr = pd.read_csv(single_file_gyr_path)
single_file_gyr.head()

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------
files = glob("../../data/raw/MetaMotion/*.csv")
len(files)
# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------
files[0].split("/")[-1]  # Get the filename
# Extract the parts of the filename
files[0].split("/")[-1].split("_")  # Split by underscore
files[0].split("/")[-1].split("_")[0].split("-")  # Split the first part by dash

# participant = files[0].split('/')[-1].split('_')[0].split('-')[0].split('\\')[-1]  # Extract participant
# label       = files[0].split('/')[-1].split('_')[0].split('-')[1]  # Extract label
# category    = files[0].split('/')[-1].split('_')[0].split('-')[2].strip('123') # Extract participant

participant = files[0].split("-")[0].replace("../../data/raw/MetaMotion\\", "")
label = files[0].split("-")[1]
category = files[0].split("-")[2].strip("123")

participant, label, category


# creating a df
df = pd.read_csv(files[0])
df["participant"] = participant
df["label"] = label
df["category"] = category
df.head()

df["participant"].value_counts()  # Check the number of participants
df["label"].value_counts()  # Check the number of labels
df["category"].value_counts()  # Check the number of categories
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------
acc_df = pd.DataFrame()
gyr_df = pd.DataFrame()

acc_set = 1
gyr_set = 1

for file in files:
    import os

    filename = os.path.basename(file)
    participant = filename.split("-")[0]
    label = filename.split("-")[1]
    category = filename.split("-")[2].split("_")[0].strip("123")

    df = pd.read_csv(file)
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in file:
        df["set"] = acc_set  # Add a set column for accelerometer data
        acc_set += 1
        acc_df = pd.concat([acc_df, df], ignore_index=True)
    elif "Gyroscope" in file:
        df["set"] = gyr_set  # Add a set column for gyroscope data
        gyr_set += 1
        gyr_df = pd.concat([gyr_df, df], ignore_index=True)

acc_df.head()
acc_df.shape

gyr_df.head()
gyr_df.shape


# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------
acc_df.info()  # Check the data types
acc_df["time (01:00)"]
pd.to_datetime(
    acc_df["time (01:00)"], format="mixed", errors="coerce"
)  # Convert to datetime

# Check for NaT values in the datetime column
pd.to_datetime(
    acc_df["time (01:00)"], format="mixed", errors="coerce"
).isnull().sum()  # Check for NaT values

pd.to_datetime(acc_df["epoch (ms)"], unit="ms", errors="coerce")  # Convert to datetime

acc_df["epoch (ms)"] = pd.to_datetime(
    acc_df["epoch (ms)"], unit="ms", errors="coerce"
)  # Convert to datetime
acc_df["time (01:00)"] = pd.to_datetime(
    acc_df["time (01:00)"], format="mixed", errors="coerce"
)  # Convert to datetime

acc_df.info()  # Check the data types again

acc_df.index = acc_df["epoch (ms)"]  # Set the index to the epoch time
acc_df = acc_df.drop(
    columns=["epoch (ms)", "time (01:00)", "elapsed (s)"]
)  # Drop the epoch column
acc_df.head()


gyr_df["epoch (ms)"] = pd.to_datetime(
    gyr_df["epoch (ms)"], unit="ms", errors="coerce"
)  # Convert to datetime
gyr_df["time (01:00)"] = pd.to_datetime(
    gyr_df["time (01:00)"], format="mixed", errors="coerce"
)  # Convert to datetime
gyr_df.info()  # Check the data types for gyroscope data
gyr_df.index = gyr_df["epoch (ms)"]  # Set the index to the epoch time
gyr_df = gyr_df.drop(
    columns=["epoch (ms)", "time (01:00)", "elapsed (s)"]
)  # Drop the epoch column
gyr_df.head()
# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------
from glob import glob

files = glob("../../data/raw/MetaMotion/*.csv")  # List all files again


def make_dataset_from_files(files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for file in files:
        participant = file.split("-")[0].replace("../../data/raw/MetaMotion\\", "")
        label = file.split("-")[1]
        category = file.split("-")[2].split("_")[0].strip("123")

        df = pd.read_csv(file)
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in file:
            df["set"] = acc_set  # Add a set column for accelerometer data
            acc_set += 1
            acc_df = pd.concat([acc_df, df], ignore_index=True)

        elif "Gyroscope" in file:
            df["set"] = gyr_set  # Add a set column for gyroscope data
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df], ignore_index=True)

    acc_df["epoch (ms)"] = pd.to_datetime(
        acc_df["epoch (ms)"], unit="ms", errors="coerce"
    )  # Convert to datetime
    acc_df["time (01:00)"] = pd.to_datetime(
        acc_df["time (01:00)"], format="mixed", errors="coerce"
    )  # Convert to datetime

    acc_df.index = acc_df["epoch (ms)"]  # Set the index to the epoch time
    acc_df = acc_df.drop(
        columns=["epoch (ms)", "time (01:00)", "elapsed (s)"]
    )  # Drop the epoch column

    gyr_df["epoch (ms)"] = pd.to_datetime(
        gyr_df["epoch (ms)"], unit="ms", errors="coerce"
    )  # Convert to datetime
    gyr_df["time (01:00)"] = pd.to_datetime(
        gyr_df["time (01:00)"], format="mixed", errors="coerce"
    )  # Convert to datetime

    gyr_df.index = gyr_df["epoch (ms)"]  # Set the index to the epoch time
    gyr_df = gyr_df.drop(
        columns=["epoch (ms)", "time (01:00)", "elapsed (s)"]
    )  # Drop the epoch column

    return acc_df, gyr_df


acc_df, gyr_df = make_dataset_from_files(files)
acc_df.iloc[:, :3].head()
gyr_df.head()


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
merged_df = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)
merged_df.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
merged_df[:1000].resample("200ms").mean(
    numeric_only=True
)  # Resample the data to 100ms intervals

agg_dict = {
    "acc_x": "mean",
    "acc_y": "mean",
    "acc_z": "mean",
    "gyr_x": "mean",
    "gyr_y": "mean",
    "gyr_z": "mean",
    "label": "last",  # Assuming label is constant within the resample period
    "category": "last",  # Assuming category is constant within the resample period
    "participant": "last",  # Assuming participant is constant within the resample period
    "set": "last",  # Assuming set is constant within the resample period
}

merged_df[:1000].resample(rule="200ms").apply(
    agg_dict
)  # Resample the data to 200ms intervals and apply aggregation

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

final_df = merged_df.resample("200ms").agg(agg_dict).dropna()
final_df["set"] = final_df["set"].astype(int)  # Ensure 'set' is of type int
final_df.head()
final_df.shape  # Check the shape of the final dataset
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
final_df.to_pickle(
    "../../data/interim/01_preprocessed_data.pkl"
)  # Save the final dataset as a pickle file
