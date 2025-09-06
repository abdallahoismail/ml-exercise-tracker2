import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.predict_model import ExercisePredictor

def debug_predictions():
    print("=== DEBUGGING EXERCISE PREDICTIONS ===")
    
    # Load real test data
    test_data = pd.read_pickle('data/interim/01_preprocessed_data.pkl')
    print(f"Test data shape: {test_data.shape}")
    print(f"Available columns: {list(test_data.columns)}")
    print(f"Unique labels: {test_data['label'].unique()}")
    
    # Test several different sets
    for s in [1, 2, 3, 4, 5]:
        if s in test_data['set'].unique():
            sensor_df = test_data[test_data['set'] == s]
            
            # Get actual label and participant
            actual_label = sensor_df['label'].iloc[0]
            participant = sensor_df['participant'].iloc[0]
            
            print(f"\n--- SET {s} ---")
            print(f"Actual: {actual_label} (Participant {participant})")
            print(f"Data shape: {sensor_df.shape}")
            
            try:
                # Make prediction
                predictor = ExercisePredictor()
                results = predictor.predict_and_count(sensor_df)
                
                predicted = results['exercise']
                reps = results['repetitions']
                
                print(f"Predicted: {predicted} ({reps} reps)")
                
                # Check if prediction is correct
                if predicted == actual_label:
                    print("✓ CORRECT")
                else:
                    print("✗ WRONG")
                    
                    # Debug feature values for wrong predictions
                    print("\n--- DEBUGGING FEATURES ---")
                    df_features = predictor.engineer_features(sensor_df)
                    
                    # Check selected features
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
                    
                    print("Selected feature values:")
                    for feature in selected_features:
                        if feature in df_features.columns:
                            values = df_features[feature].dropna()
                            if len(values) > 0:
                                print(f"  {feature}: mean={values.mean():.4f}, std={values.std():.4f}")
                            else:
                                print(f"  {feature}: NO DATA")
                        else:
                            print(f"  {feature}: MISSING")
                    
            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    debug_predictions()