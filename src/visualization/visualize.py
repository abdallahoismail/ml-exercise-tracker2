import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01_preprocessed_data.pkl")
df.head()
# --------------------------------------------------------------
# Plot single columns
# --------------------------------------------------------------
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"].reset_index(drop=True))
plt.plot(set_df["acc_x"].reset_index(drop=True))
plt.plot(set_df["acc_z"].reset_index(drop=True))


# --------------------------------------------------------------
# Plot all exercises
# --------------------------------------------------------------
plt.style.use("ggplot")
plt.rcParams["figure.figsize"] = (18, 8)
plt.rcParams["figure.dpi"] = 150

df["label"].unique()

for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


for label in df["label"].unique():
    subset = df[df["label"] == label]
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_x"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()
# --------------------------------------------------------------
# Adjust plot settings
# --------------------------------------------------------------


# --------------------------------------------------------------
# Compare medium vs. heavy sets
# --------------------------------------------------------------
category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()

category_df.groupby(["category"])["acc_y"].plot()
plt.legend()
plt.show()

# --------------------------------------------------------------
# Compare participants
# --------------------------------------------------------------
participant_df = df.query("label == 'ohp'").sort_values("participant").reset_index()

participant_df.groupby(["participant"])["acc_y"].plot()
plt.legend()
plt.show()

# --------------------------------------------------------------
# Plot multiple axis
# --------------------------------------------------------------
label = "squat"
participant = "A"
subset = (
    df.query(f"label == '{label}'")
    .query(f"participant == '{participant}'")
    .reset_index()
)

fig, ax = plt.subplots()
subset[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("Y Acceleration (g)")
ax.set_xlabel("Samples")
plt.legend(loc="upper left")
plt.show()

# --------------------------------------------------------------
# Create a loop to plot all combinations per sensor
# --------------------------------------------------------------
for label in df["label"].unique():
    for participant in df["participant"].unique():
        subset = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(subset) > 0:
            fig, ax = plt.subplots()
            subset[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            plt.title(f"{label} - {participant}")
            plt.xlabel("Samples")
            plt.ylabel("Y Acceleration (g)")
            plt.grid(True)
            plt.legend(loc="upper left")
            plt.show()


for label in df["label"].unique():
    for participant in df["participant"].unique():
        subset = (
            df.query(f"label == '{label}'")
            .query(f"participant == '{participant}'")
            .reset_index()
        )
        if len(subset) > 0:
            fig, ax = plt.subplots()
            subset[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            plt.title(f"{label} - {participant}")
            plt.xlabel("Samples")
            plt.ylabel("Angular Velocity (deg/s)")
            plt.grid(True)
            plt.legend(loc="upper left")
            plt.show()


# --------------------------------------------------------------
# Combine plots in one figure
# --------------------------------------------------------------


# --------------------------------------------------------------
# Loop over all combinations and export for both sensors
# --------------------------------------------------------------
