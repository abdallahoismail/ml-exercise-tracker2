import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from models.predict_model import ExercisePredictor


def test_production_ready_pipeline():
    """
    Test the updated prediction pipeline with pre-trained models
    """
    print("=" * 70)
    print("TESTING PRODUCTION-READY PREDICTION PIPELINE")
    print("=" * 70)

    # --- FIX START ---
    # Construct an absolute path to the data file
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        project_root, "data", "interim", "01_preprocessed_data.pkl"
    )

    # Load test data using the absolute path
    test_data = pd.read_pickle(data_path)
    # --- FIX END ---

    print(f"Test data shape: {test_data.shape}")
    print(f"Available exercises: {test_data['label'].unique()}")

    # Initialize predictor (should load pre-trained models)
    print(f"\n{'-' * 50}")
    print("INITIALIZING PREDICTOR WITH PRE-TRAINED MODELS")
    print(f"{'-' * 50}")

    try:
        predictor = ExercisePredictor()
        print("SUCCESS: Predictor initialized successfully!")
    except Exception as e:
        print(f"ERROR: Failed to initialize predictor: {e}")
        return

    # Test multiple exercises
    test_sets = [
        {"set": 1, "expected": "bench"},
        {"set": 2, "expected": "bench"},
        {"set": 3, "expected": "bench"},
        {"set": 4, "expected": "bench"},
        {"set": 5, "expected": "dead"},
        {"set": 10, "expected": "ohp"},
        {"set": 15, "expected": "ohp"},
        {"set": 20, "expected": "rest"},
        {"set": 25, "expected": "squat"},
        {"set": 30, "expected": "bench"},
    ]

    correct_predictions = 0
    total_predictions = 0

    print(f"\n{'-' * 50}")
    print("TESTING PREDICTIONS ON MULTIPLE EXERCISES")
    print(f"{'-' * 50}")

    for test_case in test_sets:
        set_num = test_case["set"]
        expected = test_case["expected"]

        # Get data for this set
        if set_num not in test_data["set"].unique():
            print(f"Set {set_num} not found, skipping...")
            continue

        sensor_df = test_data[test_data["set"] == set_num]
        actual_label = sensor_df["label"].iloc[0]
        participant = sensor_df["participant"].iloc[0]

        print(f"\n--- SET {set_num} ---")
        print(f"Expected: {expected}")
        print(f"Actual: {actual_label}")
        print(f"Participant: {participant}")
        print(f"Data shape: {sensor_df.shape}")

        try:
            # Make prediction
            results = predictor.predict_and_count(sensor_df)

            predicted = results["exercise"]
            reps = results["repetitions"]
            confidence = results["confidence"]

            print(f"Predicted: {predicted}")
            print(f"Repetitions: {reps}")
            print(f"Confidence: {confidence:.2f}")

            # Check if prediction matches actual
            is_correct = predicted == actual_label
            print(f"Result: {'CORRECT' if is_correct else 'WRONG'}")

            if is_correct:
                correct_predictions += 1
            total_predictions += 1

        except Exception as e:
            print(f"ERROR: {e}")
            total_predictions += 1

    # Calculate and display results
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n{'=' * 70}")
        print("FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"Correct predictions: {correct_predictions}/{total_predictions}")
        print(f"Accuracy: {accuracy:.2%}")

        if accuracy >= 0.8:
            print("EXCELLENT: High accuracy achieved with pre-trained models!")
        elif accuracy >= 0.6:
            print("GOOD: Significant improvement with pre-trained models!")
        elif accuracy >= 0.4:
            print("MODERATE: Some improvement with pre-trained models.")
        else:
            print("POOR: Still needs investigation.")

        print(f"{'=' * 70}")

        return accuracy
    else:
        print("No predictions were made.")
        return 0.0


def test_feature_consistency():
    """
    Test that the same input produces consistent features every time
    """
    print(f"\n{'-' * 50}")
    print("TESTING FEATURE CONSISTENCY")
    print(f"{'-' * 50}")

    # --- FIX START ---
    # Construct an absolute path to the data file
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        project_root, "data", "interim", "01_preprocessed_data.pkl"
    )

    # Load test data using the absolute path
    test_data = pd.read_pickle(data_path)
    # --- FIX END ---
    sensor_df = test_data[test_data["set"] == 1]  # Use Set 1 for consistency test

    print(f"Testing consistency with Set 1 (shape: {sensor_df.shape})")

    # Make predictions multiple times
    predictor = ExercisePredictor()

    predictions = []
    feature_samples = []

    for i in range(3):
        print(f"\nRun {i + 1}:")
        try:
            # Generate features
            df_features = predictor.engineer_features(sensor_df)

            # Make prediction
            predicted = predictor.predict_exercise(df_features)
            predictions.append(predicted)

            # Sample a few key features for comparison
            sample_features = {}
            key_features = ["pca_1", "pca_2", "pca_3", "cluster"]

            for feat in key_features:
                if feat in df_features.columns and len(df_features) > 0:
                    sample_features[feat] = df_features[feat].iloc[0]
                else:
                    sample_features[feat] = "MISSING"

            feature_samples.append(sample_features)
            print(f"  Prediction: {predicted}")
            print(f"  Sample features: {sample_features}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Check consistency
    print(f"\n{'-' * 30}")
    print("CONSISTENCY CHECK")
    print(f"{'-' * 30}")

    # Check prediction consistency
    unique_predictions = set(predictions)
    prediction_consistent = len(unique_predictions) == 1
    print(f"Predictions: {predictions}")
    print(f"Prediction consistency: {'PASS' if prediction_consistent else 'FAIL'}")

    # Check feature consistency
    if len(feature_samples) > 1:
        feature_consistent = True
        for feature_name in key_features:
            values = [
                sample[feature_name]
                for sample in feature_samples
                if feature_name in sample
            ]
            if len(set(values)) > 1:
                feature_consistent = False
                print(f"Feature {feature_name} inconsistent: {values}")

        if feature_consistent:
            print("Feature consistency: PASS")
        else:
            print("Feature consistency: FAIL")

    return prediction_consistent and (len(feature_samples) <= 1 or feature_consistent)


if __name__ == "__main__":
    print("PRODUCTION-READY PIPELINE TESTING")

    # Test 1: Overall accuracy with pre-trained models
    accuracy = test_production_ready_pipeline()

    # Test 2: Feature consistency
    consistency = test_feature_consistency()

    # Final summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Overall Accuracy: {accuracy:.2%}")
    print(f"Feature Consistency: {'PASS' if consistency else 'FAIL'}")

    if accuracy >= 0.7 and consistency:
        print("\nSUCCESS: Production pipeline is working well!")
    elif accuracy >= 0.5:
        print("\nMODERATE: Pipeline improved but needs more work.")
    else:
        print("\nNEEDS WORK: Pipeline still has issues.")

    print(f"{'=' * 70}")
