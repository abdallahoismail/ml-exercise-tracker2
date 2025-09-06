import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.predict_model_fixed import ImprovedExercisePredictor

def test_improved_predictions():
    print("=== TESTING IMPROVED EXERCISE PREDICTIONS ===")
    # Load test data
    test_data = pd.read_pickle('C:/users/abdul/desktop/repos/ml-exercise-tracker-2/data/interim/01_preprocessed_data.pkl')
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize improved predictor
    predictor = ImprovedExercisePredictor()
    
    correct_predictions = 0
    total_predictions = 0
    
    # Test several different sets
    test_sets = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    
    for s in test_sets:
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
                results = predictor.predict_and_count(sensor_df)
                
                predicted = results['exercise']
                reps = results['repetitions']
                confidence = results['confidence']
                
                print(f"Predicted: {predicted} ({reps} reps, conf={confidence:.2f})")
                
                # Check if prediction is correct
                if predicted == actual_label:
                    print("CORRECT")
                    correct_predictions += 1
                else:
                    print("WRONG")
                    
                total_predictions += 1
                
            except Exception as e:
                print(f"ERROR: {e}")
                total_predictions += 1
    
    # Calculate accuracy
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n=== RESULTS ===")
        print(f"Correct: {correct_predictions}/{total_predictions}")
        print(f"Accuracy: {accuracy:.2%}")
        
        if accuracy > 0.7:
            print("SUCCESS: Improved predictions are working well!")
        else:
            print("Still needs improvement")
    else:
        print("No predictions made")

if __name__ == "__main__":
    test_improved_predictions()