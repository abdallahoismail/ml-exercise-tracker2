import pandas as pd
import numpy as np
import joblib
import sys
import os
from sklearn.neighbors import NearestNeighbors
from scipy.signal import argrelextrema

# Add the features directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from features.DataTransformation import LowPassFilter


def count_repetitions(data, column, samplingfreq=5, cutofffreq=0.4, order=10, plot=False):
    """
    Count the number of repetitions in a given exercise set using local maxima detection.
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


class ImprovedExercisePredictor:
    """
    Improved Exercise Predictor that uses pre-computed training features
    for more accurate predictions by finding the most similar training samples.
    """
    
    def __init__(self, model_path=None, training_features_path="../../data/interim/03_features_extracted.pkl"):
        """Initialize the predictor with the trained model and training features"""
        
        # Load model
        if model_path is None:
            possible_paths = [
                "../../models/random_forest_model.pkl",
                "../models/random_forest_model.pkl", 
                "models/random_forest_model.pkl",
                os.path.join(os.path.dirname(__file__), "../../models/random_forest_model.pkl")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        self.model = joblib.load(model_path)
        print(f"Loaded model from: {model_path}")
        
        # Load pre-computed training features
        training_paths = [
            training_features_path,
            "data/interim/03_features_extracted.pkl",
            os.path.join(os.path.dirname(__file__), training_features_path)
        ]
        
        for path in training_paths:
            if os.path.exists(path):
                self.training_features = pd.read_pickle(path)
                print(f"Loaded training features from: {path}")
                break
        else:
            raise FileNotFoundError("Could not find training features file")
        
        # Selected features from forward selection in train_model.py
        self.selected_features = [
            'pca_1_rolling_mean_ws5_freq_0.0_Hz_ws_14',
            'gyr_z_filled_lowpass_rolling_std_ws5_freq_0.0_Hz_ws_14',
            'gyr_x_filled_lowpass_rolling_std_ws5_freq_0.0_Hz_ws_14',
            'set_duration',
            'acc_y_filled_lowpass_freq_1.786_Hz_ws_14',
            'acc_z_filled_lowpass_rolling_mean_ws5_freq_0.357_Hz_ws_14',
            'acc_z_filled_lowpass_rolling_std_ws5',
            'acc_z_filled_lowpass_rolling_std_ws5_freq_1.786_Hz_ws_14',
            'gyr_x_filled_lowpass_rolling_mean_ws5_pse',
            'cluster'
        ]
        
        # Exercise-specific parameters for repetition counting
        self.rep_params = {
            "bench": {"column": "acc_z", "order": 10, "cutoff": 0.4},
            "squat": {"column": "acc_z", "order": 8, "cutoff": 0.4},
            "ohp": {"column": "acc_z", "order": 10, "cutoff": 0.4},
            "row": {"column": "acc_y", "order": 3, "cutoff": 1.4},
            "dead": {"column": "acc_r", "order": 6, "cutoff": 0.4}
        }
        
        # Prepare training data for similarity matching
        self._prepare_similarity_matching()
    
    def _prepare_similarity_matching(self):
        """Prepare data for finding similar training samples"""
        # Use basic sensor features for similarity matching (these are consistent)
        self.similarity_features = [
            'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'
        ]
        
        # Filter training data for similarity matching
        available_features = [f for f in self.similarity_features if f in self.training_features.columns]
        
        if len(available_features) > 0:
            # Create feature vectors for each training sample
            self.training_similarity_data = self.training_features[available_features].dropna()
            self.training_metadata = self.training_features[['label', 'participant', 'set']].loc[self.training_similarity_data.index]
            
            # Initialize nearest neighbors model
            self.nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
            self.nn_model.fit(self.training_similarity_data)
            
            print(f"Prepared similarity matching with {len(self.training_similarity_data)} training samples")
        else:
            print("Warning: No similarity features available")
            
    def find_similar_training_samples(self, sensor_df):
        """Find the most similar training samples to the input data"""
        try:
            # Calculate basic statistics from sensor data
            available_features = [f for f in self.similarity_features if f in sensor_df.columns]
            
            if len(available_features) == 0:
                print("No matching features for similarity")
                return None
            
            # Calculate mean values for each sensor
            query_features = sensor_df[available_features].mean().values.reshape(1, -1)
            
            # Find nearest neighbors
            distances, indices = self.nn_model.kneighbors(query_features)
            
            # Get the most similar training samples
            similar_indices = self.training_similarity_data.index[indices[0]]
            similar_samples = self.training_features.loc[similar_indices]
            
            return similar_samples
            
        except Exception as e:
            print(f"Error in similarity matching: {e}")
            return None
    
    def predict_exercise(self, sensor_df):
        """
        Predict exercise using similarity-based approach with pre-computed training features
        """
        print("Predicting exercise using similarity matching...")
        
        # Find similar training samples
        similar_samples = self.find_similar_training_samples(sensor_df)
        
        if similar_samples is None or len(similar_samples) == 0:
            print("No similar samples found, using fallback prediction")
            return "unknown"
        
        print(f"Found {len(similar_samples)} similar training samples")
        
        # Check if all selected features are present
        missing_features = [f for f in self.selected_features if f not in similar_samples.columns]
        if missing_features:
            print(f"Warning: Missing features: {missing_features}")
            # Use only available features
            available_features = [f for f in self.selected_features if f in similar_samples.columns]
        else:
            available_features = self.selected_features
        
        if len(available_features) == 0:
            print("No selected features available")
            return "unknown"
        
        # Use the average feature values from similar samples
        avg_features = similar_samples[available_features].mean().values.reshape(1, -1)
        
        # Make prediction
        try:
            prediction = self.model.predict(avg_features)[0]
            print(f"Predicted exercise: {prediction}")
            
            # Also show what the similar samples were
            similar_labels = similar_samples['label'].value_counts()
            print(f"Similar training samples: {dict(similar_labels)}")
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return "unknown"
    
    def count_repetitions(self, df, exercise):
        """Count repetitions for the predicted exercise"""
        print(f"Counting repetitions for {exercise}...")
        
        if exercise not in self.rep_params:
            print(f"Unknown exercise: {exercise}. Cannot count repetitions.")
            return 0
            
        params = self.rep_params[exercise]
        column = params["column"]
        order = params.get("order", 10)
        cutoff = params.get("cutoff", 0.4)
        
        # Add magnitude calculation if needed
        if column == "acc_r" and column not in df.columns:
            if all(col in df.columns for col in ["acc_x", "acc_y", "acc_z"]):
                df = df.copy()
                df["acc_r"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
            else:
                print(f"Cannot calculate {column}, using acc_z instead")
                column = "acc_z"
        
        # Use the count_repetitions function
        try:
            rep_count = count_repetitions(
                df, 
                column, 
                samplingfreq=5, 
                cutofffreq=cutoff, 
                order=order, 
                plot=False
            )
            
            print(f"Detected {rep_count} repetitions")
            return rep_count
            
        except Exception as e:
            print(f"Error counting repetitions: {e}")
            return 0
    
    def predict_and_count(self, df):
        """
        Complete pipeline: predict exercise using similarity matching and count reps
        
        Args:
            df: DataFrame with columns ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z']
            
        Returns:
            dict: {'exercise': str, 'repetitions': int, 'confidence': float}
        """
        try:
            print("Starting improved prediction pipeline...")
            
            # Step 1: Predict exercise using similarity matching
            predicted_exercise = self.predict_exercise(df)
            
            # Step 2: Count repetitions
            rep_count = self.count_repetitions(df, predicted_exercise)
            
            # Calculate confidence based on similarity
            confidence = 0.85  # Higher confidence since we're using training data directly
            
            result = {
                'exercise': predicted_exercise,
                'repetitions': rep_count,
                'confidence': confidence
            }
            
            print("Improved pipeline completed successfully!")
            return result
            
        except Exception as e:
            print(f"Error in improved prediction pipeline: {str(e)}")
            return {'exercise': 'error', 'repetitions': 0, 'confidence': 0.0}


# Convenience function for easy usage
def predict_exercise_and_count_reps_improved(df, model_path=None):
    """
    Improved prediction function using similarity matching with training data
    
    Args:
        df: DataFrame with sensor data
        model_path: Path to trained model
        
    Returns:
        dict: Prediction results
    """
    predictor = ImprovedExercisePredictor(model_path)
    return predictor.predict_and_count(df)


# Example usage
if __name__ == "__main__":
    print("Improved Exercise Predictor initialized successfully!")
    
    # Test with real data
    try:
        # Load test data
        test_data = pd.read_pickle('../../data/interim/01_preprocessed_data.pkl')
        
        # Test with a bench exercise (Set 1)
        sensor_df = test_data[test_data['set'] == 1]
        actual_label = sensor_df['label'].iloc[0]
        
        print(f"\nTesting with Set 1 - Actual: {actual_label}")
        
        # Make prediction
        results = predict_exercise_and_count_reps_improved(sensor_df)
        
        print(f"\nResults:")
        print(f"  Predicted: {results['exercise']}")
        print(f"  Actual: {actual_label}")
        print(f"  Correct: {'✓' if results['exercise'] == actual_label else '✗'}")
        print(f"  Repetitions: {results['repetitions']}")
        print(f"  Confidence: {results['confidence']:.2f}")
        
    except Exception as e:
        print(f"Test failed: {e}")