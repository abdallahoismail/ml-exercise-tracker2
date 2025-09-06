import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.cluster import KMeans

# Add the features directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from features.DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from features.TemporalAbstraction import NumericalAbstraction
from features.FrequencyAbstraction import FourierTransformation
from features.model_persistence_guide import ProductionReadyPCA, ProductionReadyClustering
# Import count_repetitions function directly to avoid module-level code execution
from scipy.signal import argrelextrema


def count_repetitions(
    data, column, samplingfreq=5, cutofffreq=0.4, order=10, plot=False
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

    return len(peaks_indices)


class ExercisePredictor:
    """
    Complete pipeline for exercise prediction and repetition counting.
    Follows the preprocessing and feature engineering steps from build_features.py
    and uses the trained model from train_model.py
    """

    def __init__(self, model_path=None):
        if model_path is None:
            # Try different paths to find the model
            possible_paths = [
                "../../models/random_forest_model.pkl",
                "../models/random_forest_model.pkl",
                "models/random_forest_model.pkl",
                os.path.join(
                    os.path.dirname(__file__), "../../models/random_forest_model.pkl"
                ),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            if model_path is None:
                raise FileNotFoundError(
                    "Could not find random_forest_model.pkl. Please check the model file exists."
                )

        print(f"Loading model from: {model_path}")
        """Initialize the predictor with the trained model"""
        self.model = joblib.load(model_path)

        # Selected features from forward selection in train_model.py
        self.selected_features = [
            "pca_1_rolling_mean_ws5_freq_0.0_Hz_ws_14",
            "gyr_z_filled_lowpass_rolling_std_ws5_freq_0.0_Hz_ws_14",
            "gyr_x_filled_lowpass_rolling_std_ws5_freq_0.0_Hz_ws_14",
            "set_duration",
            "acc_y_filled_lowpass_freq_1.786_Hz_ws_14",
            "acc_z_filled_lowpass_rolling_mean_ws5_freq_0.357_Hz_ws_14",
            "acc_z_filled_lowpass_rolling_std_ws5",
            "acc_z_filled_lowpass_rolling_std_ws5_freq_1.786_Hz_ws_14",
            "gyr_x_filled_lowpass_rolling_mean_ws5_pse",
            "cluster",
        ]

        # Exercise-specific parameters for repetition counting
        self.rep_params = {
            "bench": {"column": "acc_z", "order": 10, "cutoff": 0.4},
            "squat": {"column": "acc_z", "order": 8, "cutoff": 0.4},
            "ohp": {"column": "acc_z", "order": 10, "cutoff": 0.4},
            "row": {"column": "acc_y", "order": 3, "cutoff": 1.4},
            "dead": {"column": "acc_r", "order": 6, "cutoff": 0.4},
        }

        # Initialize transformation classes
        self.lowpass = LowPassFilter()
        self.pca = ProductionReadyPCA()
        self.clustering = ProductionReadyClustering()
        self.num_abstractor = NumericalAbstraction()
        self.ft = FourierTransformation()
        
        # Load pre-trained models
        self._load_pretrained_models()

        # Parameters from build_features.py
        self.sampling_freq = 5  # 1000ms / 200ms = 5Hz
        self.lowpass_cutoff = 1.3
        self.lowpass_order = 5
        self.window_size = 5  # for temporal features
        self.freq_window_size = 14  # for frequency features (2.8 sec * 5 Hz)
        self.overlap = 7  # 50% overlap for frequency features

    def _load_pretrained_models(self):
        """Load pre-trained PCA and clustering models"""
        try:
            # Try different paths for model files
            pca_paths = [
                "../../models/pca_model.pkl",
                "models/pca_model.pkl",
                os.path.join(os.path.dirname(__file__), "../../models/pca_model.pkl")
            ]
            
            clustering_paths = [
                "../../models/clustering_model.pkl", 
                "models/clustering_model.pkl",
                os.path.join(os.path.dirname(__file__), "../../models/clustering_model.pkl")
            ]
            
            # Load PCA model
            pca_loaded = False
            for pca_path in pca_paths:
                if os.path.exists(pca_path):
                    self.pca.load_pca_model(pca_path)
                    pca_loaded = True
                    break
            
            if not pca_loaded:
                print("Warning: Could not find PCA model file. PCA will use default values.")
            
            # Load clustering model
            clustering_loaded = False
            for clustering_path in clustering_paths:
                if os.path.exists(clustering_path):
                    self.clustering.load_clustering_model(clustering_path)
                    clustering_loaded = True
                    break
                    
            if not clustering_loaded:
                print("Warning: Could not find clustering model file. Clustering will use default values.")
                
        except Exception as e:
            print(f"Warning: Error loading pre-trained models: {e}")
            print("Pipeline will continue but may have reduced accuracy.")

    def preprocess_data(self, df):
        """
        Apply the complete preprocessing pipeline from build_features.py
        """
        print("Starting data preprocessing...")
        df_processed = df.copy()

        # 1. Handle missing values (interpolation)
        numeric_columns = df_processed.select_dtypes(include="float").columns.tolist()

        for col in numeric_columns:
            df_processed[col + "_filled"] = df_processed[col].interpolate()

        # 2. Calculate set duration (FIX: Make calculation consistent with training)
        if "set" in df_processed.columns:
            for s in df_processed["set"].unique():
                subset = df_processed[df_processed["set"] == s]
                # Duration in seconds (timestamp difference)
                duration = (subset.index[-1] - subset.index[0]).total_seconds()
                df_processed.loc[(df_processed["set"] == s), "set_duration"] = duration
        else:
            # Default duration if no set information (use training average)
            df_processed["set_duration"] = 16.0  # Match training data

        return df_processed

    def apply_lowpass_filter(self, df):
        """Apply low-pass filter to filled columns"""
        print("Applying low-pass filter...")
        df_lowpass = df.copy()

        # Get filled columns
        filled_cols = [col for col in df_lowpass.columns if "filled" in col]

        # Apply low-pass filter
        for col in filled_cols:
            df_lowpass = self.lowpass.low_pass_filter(
                df_lowpass,
                col,
                self.sampling_freq,
                self.lowpass_cutoff,
                order=self.lowpass_order,
            )

        return df_lowpass

    def apply_pca(self, df):
        """Apply pre-trained PCA transformation to new data"""
        print("Applying pre-trained PCA...")
        df_pca = df.copy()
        
        # Get lowpass columns that were used for PCA in training
        lowpass_cols = [col for col in df_pca.columns if "lowpass" in col and not any(x in col for x in ['rolling', 'freq'])]
        
        if len(lowpass_cols) > 0 and self.pca.is_fitted:
            # Apply pre-trained PCA transformation (no fitting, just transformation)
            df_pca = self.pca.apply_pretrained_pca(df_pca, lowpass_cols)
            print(f"Applied pre-trained PCA to {len(lowpass_cols)} lowpass features")
        else:
            if not self.pca.is_fitted:
                print("Warning: PCA model not loaded. Using dummy values.")
            else:
                print("Warning: No lowpass columns found for PCA")
            # Add dummy PCA columns
            for i in range(1, 4):
                df_pca[f"pca_{i}"] = 0.0
            
        return df_pca

    def calculate_magnitude(self, df):
        """Calculate magnitude features (sum of squares)"""
        print("Calculating magnitude features...")
        df_mag = df.copy()

        # Calculate acceleration magnitude
        if all(
            col in df_mag.columns
            for col in [
                "acc_x_filled_lowpass",
                "acc_y_filled_lowpass",
                "acc_z_filled_lowpass",
            ]
        ):
            acc_r = np.sqrt(
                df_mag["acc_x_filled_lowpass"] ** 2
                + df_mag["acc_y_filled_lowpass"] ** 2
                + df_mag["acc_z_filled_lowpass"] ** 2
            )
            df_mag["acc_r"] = acc_r

        # Calculate gyroscope magnitude
        if all(
            col in df_mag.columns
            for col in [
                "gyr_x_filled_lowpass",
                "gyr_y_filled_lowpass",
                "gyr_z_filled_lowpass",
            ]
        ):
            gyr_r = np.sqrt(
                df_mag["gyr_x_filled_lowpass"] ** 2
                + df_mag["gyr_y_filled_lowpass"] ** 2
                + df_mag["gyr_z_filled_lowpass"] ** 2
            )
            df_mag["gyr_r"] = gyr_r

        return df_mag

    def apply_temporal_features(self, df):
        """Apply temporal abstraction (rolling window features)"""
        print("Applying temporal features...")
        df_temporal = df.copy()

        # Get columns for temporal features
        temporal_cols = [
            col
            for col in df_temporal.columns
            if ("lowpass" in col) or ("pca" in col) or (col.endswith("_r"))
        ]

        if len(temporal_cols) > 0:
            # If 'set' column exists, group by set to avoid mixing exercises
            if "set" in df_temporal.columns:
                grouping_cols = ["set"]
                if "participant" in df_temporal.columns:
                    grouping_cols.append("participant")
                if "label" in df_temporal.columns:
                    grouping_cols.append("label")

                # Apply rolling features grouped by set
                for col in temporal_cols:
                    df_temporal[f"{col}_rolling_mean_ws{self.window_size}"] = (
                        df_temporal.groupby(grouping_cols)[col].transform(
                            lambda x: x.rolling(
                                window=self.window_size, min_periods=self.window_size
                            ).mean()
                        )
                    )
                    df_temporal[f"{col}_rolling_std_ws{self.window_size}"] = (
                        df_temporal.groupby(grouping_cols)[col].transform(
                            lambda x: x.rolling(
                                window=self.window_size, min_periods=self.window_size
                            ).std()
                        )
                    )
            else:
                # Apply rolling features without grouping
                for col in temporal_cols:
                    df_temporal[f"{col}_rolling_mean_ws{self.window_size}"] = (
                        df_temporal[col]
                        .rolling(window=self.window_size, min_periods=self.window_size)
                        .mean()
                    )
                    df_temporal[f"{col}_rolling_std_ws{self.window_size}"] = (
                        df_temporal[col]
                        .rolling(window=self.window_size, min_periods=self.window_size)
                        .std()
                    )

        return df_temporal

    def apply_frequency_features(self, df):
        """Apply frequency domain features using FFT"""
        print("Applying frequency features...")
        df_freq = df.copy().reset_index()

        # Ensure we have an epoch column for the frequency transformation
        if "epoch (ms)" not in df_freq.columns:
            df_freq["epoch (ms)"] = df_freq.index

        # Get columns for frequency features
        freq_cols = [
            col
            for col in df_freq.columns
            if ("lowpass" in col) or ("pca" in col) or (col.endswith("_r"))
        ]

        if len(freq_cols) > 0:
            # Apply FFT-based feature extraction
            if "set" in df_freq.columns:
                # Process each set separately
                df_freq_list = []
                for s in df_freq["set"].unique():
                    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
                    if len(subset) >= self.freq_window_size:
                        try:
                            subset = self.ft.abstract_frequency(
                                subset,
                                freq_cols,
                                self.freq_window_size,
                                self.sampling_freq,
                            )
                            df_freq_list.append(subset)
                        except Exception as e:
                            print(
                                f"Warning: Error applying frequency features to set {s}: {e}"
                            )

                if df_freq_list:
                    df_freq = pd.concat(df_freq_list)
                    # Try to set index, but don't fail if epoch column doesn't exist
                    if "epoch (ms)" in df_freq.columns:
                        df_freq = df_freq.set_index("epoch (ms)", drop=True)
                else:
                    # If no sets have enough data, try whole dataframe
                    if len(df_freq) >= self.freq_window_size:
                        try:
                            df_freq = self.ft.abstract_frequency(
                                df_freq,
                                freq_cols,
                                self.freq_window_size,
                                self.sampling_freq,
                            )
                            if "epoch (ms)" in df_freq.columns:
                                df_freq = df_freq.set_index("epoch (ms)", drop=True)
                        except Exception as e:
                            print(f"Warning: Error applying frequency features: {e}")
            else:
                # Apply to whole dataframe
                if len(df_freq) >= self.freq_window_size:
                    try:
                        df_freq = self.ft.abstract_frequency(
                            df_freq,
                            freq_cols,
                            self.freq_window_size,
                            self.sampling_freq,
                        )
                        if "epoch (ms)" in df_freq.columns:
                            df_freq = df_freq.set_index("epoch (ms)", drop=True)
                    except Exception as e:
                        print(f"Warning: Error applying frequency features: {e}")

        return df_freq

    def apply_clustering(self, df):
        """Apply pre-trained clustering to new data"""
        print("Applying pre-trained clustering...")
        df_cluster = df.copy()
        
        # Cluster columns (accelerometer lowpass features) - same as training
        cluster_cols = ["acc_x_filled_lowpass", "acc_y_filled_lowpass", "acc_z_filled_lowpass"]
        
        if all(col in df_cluster.columns for col in cluster_cols) and self.clustering.is_fitted:
            # Apply pre-trained clustering transformation (no fitting, just prediction)
            df_cluster = self.clustering.apply_pretrained_clustering(df_cluster, cluster_cols)
            print(f"Applied pre-trained clustering")
        else:
            if not self.clustering.is_fitted:
                print("Warning: Clustering model not loaded. Using default cluster 0")
            else:
                print("Warning: Missing cluster input columns, using default cluster 0")
            # Default cluster if model not loaded or columns don't exist
            df_cluster["cluster"] = 0
            
        return df_cluster

    def engineer_features(self, df):
        """
        Apply the complete feature engineering pipeline
        """
        print("Starting feature engineering pipeline...")

        # Step 1: Preprocess data
        df_processed = self.preprocess_data(df)

        # Step 2: Apply low-pass filter
        df_lowpass = self.apply_lowpass_filter(df_processed)

        # Step 3: Apply PCA
        df_pca = self.apply_pca(df_lowpass)

        # Step 4: Calculate magnitude features
        df_mag = self.calculate_magnitude(df_pca)

        # Step 5: Apply temporal features
        df_temporal = self.apply_temporal_features(df_mag)

        # Step 6: Apply frequency features
        df_freq = self.apply_frequency_features(df_temporal)

        # Step 7: Apply clustering
        df_cluster = self.apply_clustering(df_freq)

        # Step 8: Handle overlapping windows (take every second row)
        df_final = df_cluster.iloc[::2, :].copy()

        # Drop NaN values
        df_final = df_final.dropna()

        print("Feature engineering completed!")
        return df_final

    def predict_exercise(self, df):
        """
        Predict exercise type using the trained model
        """
        print("Predicting exercise...")

        # Check if all selected features are present
        missing_features = [f for f in self.selected_features if f not in df.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Fill missing features with zeros or mean values
            for feature in missing_features:
                df[feature] = 0

        # Make predictions using only selected features
        predictions = self.model.predict(df[self.selected_features])

        # Return the most common prediction
        unique, counts = np.unique(predictions, return_counts=True)
        predicted_exercise = unique[np.argmax(counts)]

        print(f"Predicted exercise: {predicted_exercise}")
        return predicted_exercise

    def count_repetitions(self, df, exercise):
        """
        Count repetitions for the predicted exercise
        """
        print(f"Counting repetitions for {exercise}...")

        if exercise not in self.rep_params:
            print(f"Unknown exercise: {exercise}. Cannot count repetitions.")
            return 0

        params = self.rep_params[exercise]
        column = params["column"]
        order = params["order"]
        cutoff = params.get("cutoff", 0.4)

        # Use the count_repetitions function
        rep_count = count_repetitions(
            df,
            column,
            samplingfreq=self.sampling_freq,
            cutofffreq=cutoff,
            order=order,
            plot=False,
        )

        print(f"Detected {rep_count} repetitions")
        return rep_count

    def predict_and_count(self, df):
        """
        Complete pipeline: preprocess, engineer features, predict exercise, and count reps

        Args:
            df: DataFrame with columns ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
            Should have DatetimeIndex for timestamp information

        Returns:
            dict: {'exercise': str, 'repetitions': int, 'confidence': float}
        """
        try:
            print("Starting complete prediction pipeline...")

            # Step 1: Engineer features
            df_features = self.engineer_features(df)

            if len(df_features) == 0:
                print("No data remaining after feature engineering")
                return {"exercise": "unknown", "repetitions": 0, "confidence": 0.0}

            # Step 2: Predict exercise
            predicted_exercise = self.predict_exercise(df_features)

            # Step 3: Count repetitions using original data (before heavy processing)
            # Add magnitude calculation for original data if needed
            df_for_reps = df.copy()
            if "acc_r" not in df_for_reps.columns:
                df_for_reps["acc_r"] = np.sqrt(
                    df_for_reps["acc_x"] ** 2
                    + df_for_reps["acc_y"] ** 2
                    + df_for_reps["acc_z"] ** 2
                )

            rep_count = self.count_repetitions(df_for_reps, predicted_exercise)

            # Calculate confidence (could be enhanced with probability scores)
            confidence = 0.8  # Placeholder confidence

            result = {
                "exercise": predicted_exercise,
                "repetitions": rep_count,
                "confidence": confidence,
            }

            print("Pipeline completed successfully!")
            return result

        except Exception as e:
            print(f"Error in prediction pipeline: {str(e)}")
            return {"exercise": "error", "repetitions": 0, "confidence": 0.0}


# Convenience function for easy usage
def predict_exercise_and_count_reps(
    df, model_path="../../models/random_forest_model.pkl"
):
    """
    Convenience function to predict exercise and count repetitions

    Args:
        df: DataFrame with sensor data
        model_path: Path to trained model

    Returns:
        dict: Prediction results
    """
    predictor = ExercisePredictor(model_path)
    return predictor.predict_and_count(df)


# Example usage
if __name__ == "__main__":
    # Example usage with sample data
    print("Exercise Predictor initialized successfully!")

    # Example with real data usage:
    # 1. Load your sensor data (must have columns: acc_x, acc_y,
    # acc_z, gyr_x, gyr_y, gyr_z)
    # df = pd.read_csv("your_sensor_data.csv", index_col=0, parse_dates=True)
    #
    # 2. Make prediction
    # results = predict_exercise_and_count_reps(df)
    # print(f"Exercise: {results['exercise']}")
    # print(f"Repetitions: {results['repetitions']}")
    # print(f"Confidence: {results['confidence']}")

    # Create demonstration with synthetic data
    print("\nCreating demonstration with synthetic data...")
    n_samples = 50  # 10 seconds at 5Hz
    timestamps = pd.date_range(
        start="2023-01-01 10:00:00", periods=n_samples, freq="200ms"
    )

    # Simulate squat exercise (pattern in acc_z)
    data = {
        "acc_x": np.random.normal(0, 0.3, n_samples),
        "acc_y": np.random.normal(0, 0.2, n_samples),
        "acc_z": 8
        + 3 * np.sin(np.linspace(0, 8 * np.pi, n_samples))
        + np.random.normal(0, 0.3, n_samples),
        "gyr_x": np.random.normal(0, 15, n_samples),
        "gyr_y": np.random.normal(0, 12, n_samples),
        "gyr_z": 20 * np.sin(np.linspace(0, 8 * np.pi, n_samples))
        + np.random.normal(0, 5, n_samples),
    }

    demo_df = pd.DataFrame(data, index=timestamps)

    try:
        results = predict_exercise_and_count_reps(demo_df)
        print("\nDemo Results:")
        print(f"  Exercise: {results['exercise']}")
        print(f"  Repetitions: {results['repetitions']}")
        print(f"  Confidence: {results['confidence']:.2f}")
        print("\nDemo completed successfully!")
    except Exception as e:
        print(f"Demo failed: {e}")
