# Simple Testing Guide for ML Exercise Tracker Pipeline

## Overview
You have 3 test files available. Here's how to use them step by step:

## Method 1: Command Line Testing (Recommended)

### Option A: Use test_functions.py (Most Robust)
```bash
cd C:\Users\abdul\Desktop\Repos\ml-exercise-tracker-2
python test_functions.py
```

### Option B: Use test_production_pipeline.py
```bash
cd C:\Users\abdul\Desktop\Repos\ml-exercise-tracker-2
python test_production_pipeline.py
```

## Method 2: Interactive Testing

### Quick Single Test
```python
# In Python interpreter or Jupyter
import sys
sys.path.append(r'C:\Users\abdul\Desktop\Repos\ml-exercise-tracker-2')

from test_functions import quick_test
result = quick_test()
```

### Full Test Suite
```python
from test_functions import test_production_ready_pipeline, test_feature_consistency

# Test accuracy
accuracy = test_production_ready_pipeline()
print(f"Accuracy: {accuracy:.2%}")

# Test consistency
consistency = test_feature_consistency()
print(f"Consistent: {consistency}")
```

## Method 3: Jupyter Notebook Testing

```python
# Import notebook-friendly functions
from notebook_tests import test_single_prediction, test_accuracy, run_all_tests

# Quick single test
result = test_single_prediction()

# Full accuracy test
accuracy = test_accuracy()

# Complete test suite
all_results = run_all_tests()
```

## What Each Test Does

### 1. `test_single_prediction()` 
- Tests prediction on one exercise set
- Shows actual vs predicted exercise
- Quick verification that pipeline works

### 2. `test_accuracy()`
- Tests multiple exercise sets
- Calculates overall accuracy percentage
- Shows which predictions are correct/wrong

### 3. `test_consistency()`
- Runs same data through pipeline multiple times
- Verifies identical results each time
- Confirms PCA/clustering models are working correctly

## Expected Results

### Good Results:
- **Accuracy**: 80-100% correct predictions
- **Consistency**: All runs produce identical results
- **No errors**: Pipeline completes without crashes

### What Success Looks Like:
```
Accuracy: 100%
Feature consistency: PASS
Predictions: ['bench', 'bench', 'bench'] ✓ CONSISTENT
Overall Status: ✓ SUCCESS
```

## Troubleshooting

### If you get "FileNotFoundError":
1. Make sure you're in the project directory: `C:\Users\abdul\Desktop\Repos\ml-exercise-tracker-2`
2. Check if the data file exists: `data\interim\01_preprocessed_data.pkl`
3. Use the absolute path version in test_functions.py (most robust)

### If imports fail:
```python
import sys
sys.path.append(r'C:\Users\abdul\Desktop\Repos\ml-exercise-tracker-2\src')
```

### If models not found:
- Make sure these exist:
  - `models\random_forest_model.pkl`
  - `models\pca_model.pkl` 
  - `models\clustering_model.pkl`

## Quick Commands Summary

```bash
# Navigate to project
cd C:\Users\abdul\Desktop\Repos\ml-exercise-tracker-2

# Run most robust test
python test_functions.py

# Or run specific test
python -c "from test_functions import quick_test; quick_test()"

# For Jupyter users
python -c "from notebook_tests import run_all_tests; run_all_tests()"
```

## What Files Are Tested

The tests use your actual training data from `data\interim\01_preprocessed_data.pkl` and test on multiple exercise sets:
- Set 1-4: Bench press
- Set 5: Deadlift  
- Set 10, 15: Overhead press
- Set 20: Rest
- Set 25: Squat
- Set 30: Bench press

## Success Criteria

✅ **Pipeline Working**: Accuracy ≥ 80% and consistency = PASS
✅ **Production Ready**: No FileNotFound or import errors
✅ **Model Integration**: PCA and clustering models load successfully

Run `python test_functions.py` to get started!