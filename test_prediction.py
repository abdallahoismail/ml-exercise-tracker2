import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.predict_model import ExercisePredictor

def create_sample_data():
    """Create sample sensor data for testing"""
    # Create 5 seconds of data at 5Hz (25 samples)
    n_samples = 25
    timestamps = pd.date_range(start='2023-01-01 10:00:00', periods=n_samples, freq='200ms')
    
    # Simulate bench press data (oscillating pattern in acc_z)
    data = {
        'acc_x': np.random.normal(0, 0.5, n_samples),
        'acc_y': np.random.normal(0, 0.3, n_samples), 
        'acc_z': 5 * np.sin(np.linspace(0, 4*np.pi, n_samples)) + np.random.normal(0, 0.2, n_samples),
        'gyr_x': np.random.normal(0, 10, n_samples),
        'gyr_y': np.random.normal(0, 8, n_samples),
        'gyr_z': np.random.normal(0, 12, n_samples),
        'participant': ['A'] * n_samples,
        'label': ['bench'] * n_samples,  # Optional for testing
        'category': ['heavy'] * n_samples,  # Optional for testing 
        'set': [1] * n_samples  # Optional for testing
    }
    
    df = pd.DataFrame(data, index=timestamps)
    return df

def test_prediction_pipeline():
    """Test the complete prediction pipeline"""
    try:
        print("Testing Exercise Prediction Pipeline")
        print("=" * 40)
        
        # Create sample data
        print("Creating sample data...")
        df = create_sample_data()
        print(f"Sample data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Initialize predictor
        print("\nInitializing predictor...")
        try:
            predictor = ExercisePredictor()
            print("SUCCESS: Predictor initialized successfully!")
        except Exception as e:
            print(f"ERROR: Error initializing predictor: {e}")
            print("Make sure the trained model exists at models/random_forest_model.pkl")
            return
        
        # Test feature engineering only
        print("\nTesting feature engineering...")
        try:
            df_features = predictor.engineer_features(df)
            print(f"SUCCESS: Feature engineering completed! Output shape: {df_features.shape}")
            print(f"Number of features created: {len(df_features.columns)}")
        except Exception as e:
            print(f"ERROR: Error in feature engineering: {e}")
            return
            
        # Test full pipeline
        print("\nTesting complete prediction pipeline...")
        try:
            results = predictor.predict_and_count(df)
            print("SUCCESS: Complete pipeline executed successfully!")
            print("\nResults:")
            print(f"  Exercise: {results['exercise']}")
            print(f"  Repetitions: {results['repetitions']}")
            print(f"  Confidence: {results['confidence']}")
            
        except Exception as e:
            print(f"ERROR: Error in prediction pipeline: {e}")
            return
            
        print("\n" + "=" * 40)
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")

if __name__ == "__main__":
    test_prediction_pipeline()