"""
Test functions designed for use in Jupyter notebooks.
Import this file and use the functions directly.

Usage in Jupyter:
    from notebook_tests import test_single_prediction, test_accuracy, test_consistency
    
    # Quick test
    result = test_single_prediction()
    
    # Full tests
    accuracy = test_accuracy()
    consistency = test_consistency()
"""

import pandas as pd
import numpy as np
import sys
import os

# Auto-detect project root and add to path
def setup_project_path():
    """Automatically find and add project path"""
    current_dir = os.getcwd()
    
    # Look for src directory in current or parent directories
    search_paths = [
        current_dir,
        os.path.dirname(current_dir),
        os.path.join(current_dir, '..'),
        'C:/Users/abdul/Desktop/Repos/ml-exercise-tracker-2',  # Fallback
    ]
    
    for path in search_paths:
        src_path = os.path.join(path, 'src')
        if os.path.exists(src_path):
            if src_path not in sys.path:
                sys.path.append(src_path)
            return path
    
    raise FileNotFoundError("Could not find project src directory")

# Setup paths
try:
    PROJECT_ROOT = setup_project_path()
    from models.predict_model import ExercisePredictor
    print(f"✓ Successfully imported from project at: {PROJECT_ROOT}")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're running from the project directory or notebook is in the right location")

def find_data_file():
    """Find the preprocessed data file"""
    possible_paths = [
        os.path.join(PROJECT_ROOT, 'data', 'interim', '01_preprocessed_data.pkl'),
        'data/interim/01_preprocessed_data.pkl',
        '../data/interim/01_preprocessed_data.pkl',
        '../../data/interim/01_preprocessed_data.pkl',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find data file. Searched: {possible_paths}")

def test_single_prediction(set_num=1):
    """
    Test a single prediction - quick and simple
    
    Args:
        set_num: Which set to test (default: 1)
        
    Returns:
        dict: Results with prediction info
    """
    print(f"Testing single prediction on Set {set_num}")
    print("-" * 40)
    
    try:
        # Load data
        data_path = find_data_file()
        test_data = pd.read_pickle(data_path)
        
        if set_num not in test_data['set'].unique():
            available_sets = sorted(test_data['set'].unique())[:10]
            print(f"Set {set_num} not found. Available sets: {available_sets}")
            set_num = available_sets[0]
        
        sensor_df = test_data[test_data['set'] == set_num]
        actual_label = sensor_df['label'].iloc[0]
        participant = sensor_df['participant'].iloc[0]
        
        print(f"Set {set_num} - Participant {participant}")
        print(f"Actual exercise: {actual_label}")
        print(f"Data shape: {sensor_df.shape}")
        
        # Make prediction
        predictor = ExercisePredictor()
        results = predictor.predict_and_count(sensor_df)
        
        predicted = results['exercise']
        reps = results['repetitions']
        confidence = results['confidence']
        is_correct = predicted == actual_label
        
        print(f"\nResults:")
        print(f"  Predicted: {predicted}")
        print(f"  Repetitions: {reps}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Correct: {'✓ YES' if is_correct else '✗ NO'}")
        
        return {
            'set': set_num,
            'actual': actual_label,
            'predicted': predicted,
            'repetitions': reps,
            'confidence': confidence,
            'correct': is_correct,
            'participant': participant
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}

def test_accuracy(num_sets=5):
    """
    Test accuracy across multiple sets
    
    Args:
        num_sets: Number of sets to test (default: 5)
        
    Returns:
        float: Accuracy as percentage (0.0 to 1.0)
    """
    print(f"Testing accuracy on {num_sets} sets")
    print("=" * 50)
    
    try:
        # Load data
        data_path = find_data_file()
        test_data = pd.read_pickle(data_path)
        
        # Get test sets
        available_sets = sorted(test_data['set'].unique())[:num_sets]
        
        correct = 0
        total = 0
        results = []
        
        predictor = ExercisePredictor()
        
        for set_num in available_sets:
            sensor_df = test_data[test_data['set'] == set_num]
            actual_label = sensor_df['label'].iloc[0]
            
            try:
                prediction_result = predictor.predict_and_count(sensor_df)
                predicted = prediction_result['exercise']
                
                is_correct = predicted == actual_label
                if is_correct:
                    correct += 1
                
                results.append({
                    'set': set_num,
                    'actual': actual_label,
                    'predicted': predicted,
                    'correct': is_correct
                })
                
                print(f"Set {set_num:2d}: {actual_label:5s} → {predicted:5s} {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                print(f"Set {set_num:2d}: ERROR - {e}")
                
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        print("\n" + "=" * 50)
        print(f"Results: {correct}/{total} correct")
        print(f"Accuracy: {accuracy:.1%}")
        
        return accuracy
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 0.0

def test_consistency(set_num=1, num_runs=3):
    """
    Test feature consistency across multiple runs
    
    Args:
        set_num: Which set to test (default: 1)
        num_runs: Number of runs to test (default: 3)
        
    Returns:
        bool: True if consistent, False otherwise
    """
    print(f"Testing consistency on Set {set_num} ({num_runs} runs)")
    print("-" * 50)
    
    try:
        # Load data
        data_path = find_data_file()
        test_data = pd.read_pickle(data_path)
        sensor_df = test_data[test_data['set'] == set_num]
        
        predictor = ExercisePredictor()
        
        predictions = []
        pca_values = []
        cluster_values = []
        
        for i in range(num_runs):
            # Make prediction
            results = predictor.predict_and_count(sensor_df)
            predictions.append(results['exercise'])
            
            # Get feature values
            df_features = predictor.engineer_features(sensor_df)
            if len(df_features) > 0:
                pca_values.append({
                    'pca_1': df_features['pca_1'].iloc[0] if 'pca_1' in df_features.columns else None,
                    'pca_2': df_features['pca_2'].iloc[0] if 'pca_2' in df_features.columns else None,
                    'pca_3': df_features['pca_3'].iloc[0] if 'pca_3' in df_features.columns else None,
                })
                cluster_values.append(df_features['cluster'].iloc[0] if 'cluster' in df_features.columns else None)
            
            print(f"Run {i+1}: {results['exercise']}")
        
        # Check consistency
        predictions_consistent = len(set(predictions)) == 1
        pca_consistent = len(set(str(pca) for pca in pca_values)) == 1 if pca_values else True
        cluster_consistent = len(set(cluster_values)) == 1 if cluster_values else True
        
        overall_consistent = predictions_consistent and pca_consistent and cluster_consistent
        
        print("\nConsistency Check:")
        print(f"  Predictions: {'✓ PASS' if predictions_consistent else '✗ FAIL'} - {predictions}")
        print(f"  PCA values:  {'✓ PASS' if pca_consistent else '✗ FAIL'}")
        print(f"  Clusters:    {'✓ PASS' if cluster_consistent else '✗ FAIL'} - {cluster_values}")
        print(f"  Overall:     {'✓ PASS' if overall_consistent else '✗ FAIL'}")
        
        return overall_consistent
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def run_all_tests():
    """
    Run all tests and provide summary
    
    Returns:
        dict: Summary of all test results
    """
    print("RUNNING ALL TESTS")
    print("=" * 60)
    
    # Test 1: Single prediction
    print("\n1. SINGLE PREDICTION TEST")
    single_result = test_single_prediction()
    single_success = 'error' not in single_result
    
    # Test 2: Accuracy
    print("\n2. ACCURACY TEST")
    accuracy = test_accuracy()
    accuracy_good = accuracy >= 0.8
    
    # Test 3: Consistency
    print("\n3. CONSISTENCY TEST")
    consistency = test_consistency()
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Single Prediction: {'✓ PASS' if single_success else '✗ FAIL'}")
    print(f"Accuracy:         {'✓ PASS' if accuracy_good else '✗ FAIL'} ({accuracy:.1%})")
    print(f"Consistency:      {'✓ PASS' if consistency else '✗ FAIL'}")
    
    overall_success = single_success and accuracy_good and consistency
    print(f"\nOverall Status:   {'✓ SUCCESS' if overall_success else '✗ NEEDS WORK'}")
    
    return {
        'single_prediction': single_success,
        'accuracy': accuracy,
        'consistency': consistency,
        'overall_success': overall_success
    }

# Quick usage examples
if __name__ == "__main__":
    print("Available functions:")
    print("- test_single_prediction(set_num=1)")
    print("- test_accuracy(num_sets=5)")
    print("- test_consistency(set_num=1, num_runs=3)")
    print("- run_all_tests()")
    
    print("\nRunning quick test...")
    run_all_tests()