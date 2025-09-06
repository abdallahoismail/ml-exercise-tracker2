import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.predict_model import ExercisePredictor

def compare_features():
    print("=== COMPARING TRAINING VS PREDICTION FEATURES ===")
    
    # Load original processed training data
    try:
        training_features = pd.read_pickle('data/interim/03_features_extracted.pkl')
        print(f"Training data shape: {training_features.shape}")
        print(f"Training data columns: {len(training_features.columns)}")
    except FileNotFoundError:
        print("Training feature data not found")
        return
    
    # Load test data and generate features with our pipeline
    test_data = pd.read_pickle('data/interim/01_preprocessed_data.pkl')
    
    # Test with one specific set (bench exercise)
    s = 1
    sensor_df = test_data[test_data['set'] == s]
    actual_label = sensor_df['label'].iloc[0]
    
    print(f"\nTesting Set {s} - Actual: {actual_label}")
    
    # Generate features using our pipeline
    predictor = ExercisePredictor()
    prediction_features = predictor.engineer_features(sensor_df)
    
    print(f"Prediction features shape: {prediction_features.shape}")
    print(f"Prediction features columns: {len(prediction_features.columns)}")
    
    # Get the corresponding training features for comparison
    training_subset = training_features[
        (training_features['set'] == s) & 
        (training_features['label'] == actual_label)
    ]
    
    if len(training_subset) > 0:
        print(f"Training subset shape: {training_subset.shape}")
        
        # Compare selected features
        selected_features = [
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
        
        print("\n=== FEATURE COMPARISON ===")
        for feature in selected_features:
            
            # Training features
            if feature in training_subset.columns:
                train_vals = training_subset[feature].dropna()
                train_mean = train_vals.mean() if len(train_vals) > 0 else np.nan
                train_std = train_vals.std() if len(train_vals) > 0 else np.nan
            else:
                train_mean = train_std = "MISSING"
            
            # Prediction features  
            if feature in prediction_features.columns:
                pred_vals = prediction_features[feature].dropna()
                pred_mean = pred_vals.mean() if len(pred_vals) > 0 else np.nan
                pred_std = pred_vals.std() if len(pred_vals) > 0 else np.nan
            else:
                pred_mean = pred_std = "MISSING"
            
            print(f"\n{feature}:")
            print(f"  Training: mean={train_mean:.4f}, std={train_std:.4f}" if isinstance(train_mean, float) else f"  Training: {train_mean}")
            print(f"  Prediction: mean={pred_mean:.4f}, std={pred_std:.4f}" if isinstance(pred_mean, float) else f"  Prediction: {pred_mean}")
            
            # Check if values are significantly different
            if isinstance(train_mean, float) and isinstance(pred_mean, float):
                if abs(train_mean - pred_mean) > 0.1 or abs(train_std - pred_std) > 0.1:
                    print(f"  *** SIGNIFICANT DIFFERENCE ***")
    else:
        print("No matching training data found")
        
    # Check column differences
    training_cols = set(training_features.columns)
    prediction_cols = set(prediction_features.columns)
    
    missing_in_prediction = training_cols - prediction_cols
    extra_in_prediction = prediction_cols - training_cols
    
    print(f"\n=== COLUMN DIFFERENCES ===")
    print(f"Missing in prediction ({len(missing_in_prediction)}): {list(missing_in_prediction)[:10]}")
    print(f"Extra in prediction ({len(extra_in_prediction)}): {list(extra_in_prediction)[:10]}")
    
if __name__ == "__main__":
    compare_features()