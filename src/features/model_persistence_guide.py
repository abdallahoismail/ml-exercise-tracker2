"""
Complete Guide: PCA and Clustering Model Persistence for Training and Production

This file demonstrates how to properly save and load PCA and clustering models
to ensure consistency between training and prediction phases.

Author: ML Exercise Tracker Team
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import copy


class ProductionReadyPCA:
    """
    Enhanced PCA class that supports model persistence for production use.
    
    Key Features:
    - Fit PCA during training and save the model
    - Load pre-trained PCA model for predictions
    - Consistent normalization between training and prediction
    - Transform-only mode for production
    """
    
    def __init__(self):
        self.pca = None
        self.normalization_params = {}
        self.is_fitted = False
        
    def normalize_dataset(self, data_table, columns, save_params=False):
        """
        Normalize dataset using min-max normalization.
        
        Args:
            data_table: DataFrame to normalize
            columns: Columns to normalize
            save_params: If True, save normalization parameters for later use
            
        Returns:
            Normalized DataFrame
        """
        dt_norm = copy.deepcopy(data_table)
        
        for col in columns:
            if save_params:
                # Save normalization parameters during training
                mean_val = data_table[col].mean()
                min_val = data_table[col].min()
                max_val = data_table[col].max()
                
                self.normalization_params[col] = {
                    'mean': mean_val,
                    'min': min_val, 
                    'max': max_val,
                    'range': max_val - min_val
                }
                
                # Apply normalization
                dt_norm[col] = (data_table[col] - mean_val) / (max_val - min_val)
                
            else:
                # Use saved parameters during prediction
                if col in self.normalization_params:
                    params = self.normalization_params[col]
                    dt_norm[col] = (data_table[col] - params['mean']) / params['range']
                else:
                    raise ValueError(f"No normalization parameters found for column {col}")
                    
        return dt_norm

    def fit_and_apply_pca(self, data_table, cols, number_comp):
        """
        Fit PCA on training data and apply transformation.
        USE THIS METHOD DURING TRAINING ONLY.
        
        Args:
            data_table: Training DataFrame
            cols: Columns to apply PCA on
            number_comp: Number of PCA components
            
        Returns:
            DataFrame with PCA columns added
        """
        print(f"[TRAINING] Fitting PCA with {number_comp} components on {len(cols)} features...")
        
        # Normalize the data and save parameters for later use
        dt_norm = self.normalize_dataset(data_table, cols, save_params=True)
        
        # Fit PCA on normalized training data
        self.pca = PCA(n_components=number_comp, random_state=42)
        self.pca.fit(dt_norm[cols])
        self.is_fitted = True
        
        # Transform the training data
        new_values = self.pca.transform(dt_norm[cols])
        
        # Add PCA columns to original DataFrame
        for comp in range(number_comp):
            data_table[f"pca_{comp + 1}"] = new_values[:, comp]
            
        print(f"[TRAINING] PCA fitted successfully. Explained variance: {self.pca.explained_variance_ratio_}")
        return data_table
    
    def apply_pretrained_pca(self, data_table, cols):
        """
        Apply pre-trained PCA transformation to new data.
        USE THIS METHOD DURING PREDICTION ONLY.
        
        Args:
            data_table: New DataFrame to transform
            cols: Columns to apply PCA on (must match training)
            
        Returns:
            DataFrame with PCA columns added
        """
        if not self.is_fitted:
            raise ValueError("PCA model is not fitted. Load a pre-trained model first.")
            
        print("[PREDICTION] Applying pre-trained PCA to new data...")
        
        # Normalize using saved parameters (no fitting)
        dt_norm = self.normalize_dataset(data_table, cols, save_params=False)
        
        # Transform using pre-fitted PCA
        new_values = self.pca.transform(dt_norm[cols])
        
        # Add PCA columns to original DataFrame
        for comp in range(self.pca.n_components_):
            data_table[f"pca_{comp + 1}"] = new_values[:, comp]
            
        print("[PREDICTION] PCA transformation applied successfully.")
        return data_table
    
    def save_pca_model(self, filepath):
        """
        Save the fitted PCA model and normalization parameters.
        
        Args:
            filepath: Path to save the model (e.g., "models/pca_model.pkl")
        """
        if not self.is_fitted:
            raise ValueError("No PCA model to save. Fit the model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save both PCA model and normalization parameters
        model_data = {
            'pca_model': self.pca,
            'normalization_params': self.normalization_params,
            'is_fitted': self.is_fitted,
            'n_components': self.pca.n_components_,
            'explained_variance_ratio': self.pca.explained_variance_ratio_
        }
        
        joblib.dump(model_data, filepath)
        print(f"[SAVE] PCA model saved to: {filepath}")
        
    def load_pca_model(self, filepath):
        """
        Load a pre-trained PCA model and normalization parameters.
        
        Args:
            filepath: Path to the saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"PCA model file not found: {filepath}")
            
        # Load the saved model data
        model_data = joblib.load(filepath)
        
        self.pca = model_data['pca_model']
        self.normalization_params = model_data['normalization_params']
        self.is_fitted = model_data['is_fitted']
        
        print(f"[LOAD] PCA model loaded from: {filepath}")
        print(f"[LOAD] Components: {model_data['n_components']}")
        print(f"[LOAD] Explained variance: {model_data['explained_variance_ratio']}")


class ProductionReadyClustering:
    """
    Enhanced clustering class that supports model persistence for production use.
    
    Key Features:
    - Fit clustering during training and save the model  
    - Load pre-trained clustering model for predictions
    - Consistent cluster assignments between training and prediction
    """
    
    def __init__(self):
        self.kmeans = None
        self.is_fitted = False
        self.n_clusters = None
        
    def fit_and_apply_clustering(self, data_table, cols, n_clusters=4):
        """
        Fit clustering on training data and apply transformation.
        USE THIS METHOD DURING TRAINING ONLY.
        
        Args:
            data_table: Training DataFrame
            cols: Columns to apply clustering on
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster column added
        """
        print(f"[TRAINING] Fitting K-Means clustering with {n_clusters} clusters on {len(cols)} features...")
        
        # Fit K-means on training data
        self.kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
        cluster_assignments = self.kmeans.fit_predict(data_table[cols])
        
        # Add cluster column to original DataFrame
        data_table["cluster"] = cluster_assignments
        
        self.is_fitted = True
        self.n_clusters = n_clusters
        
        print("[TRAINING] K-Means fitted successfully.")
        print(f"[TRAINING] Cluster distribution: {pd.Series(cluster_assignments).value_counts().sort_index().to_dict()}")
        
        return data_table
    
    def apply_pretrained_clustering(self, data_table, cols):
        """
        Apply pre-trained clustering to new data.
        USE THIS METHOD DURING PREDICTION ONLY.
        
        Args:
            data_table: New DataFrame to cluster
            cols: Columns to apply clustering on (must match training)
            
        Returns:
            DataFrame with cluster column added
        """
        if not self.is_fitted:
            raise ValueError("Clustering model is not fitted. Load a pre-trained model first.")
            
        print("[PREDICTION] Applying pre-trained K-Means clustering to new data...")
        
        # Apply clustering using pre-fitted model
        cluster_assignments = self.kmeans.predict(data_table[cols])
        data_table["cluster"] = cluster_assignments
        
        print(f"[PREDICTION] Clustering applied successfully.")
        print(f"[PREDICTION] Cluster distribution: {pd.Series(cluster_assignments).value_counts().sort_index().to_dict()}")
        
        return data_table
    
    def save_clustering_model(self, filepath):
        """
        Save the fitted clustering model.
        
        Args:
            filepath: Path to save the model (e.g., "models/clustering_model.pkl")
        """
        if not self.is_fitted:
            raise ValueError("No clustering model to save. Fit the model first.")
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save clustering model
        model_data = {
            'kmeans_model': self.kmeans,
            'is_fitted': self.is_fitted,
            'n_clusters': self.n_clusters,
            'cluster_centers': self.kmeans.cluster_centers_,
            'inertia': self.kmeans.inertia_
        }
        
        joblib.dump(model_data, filepath)
        print(f"[SAVE] Clustering model saved to: {filepath}")
        
    def load_clustering_model(self, filepath):
        """
        Load a pre-trained clustering model.
        
        Args:
            filepath: Path to the saved model file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Clustering model file not found: {filepath}")
            
        # Load the saved model data
        model_data = joblib.load(filepath)
        
        self.kmeans = model_data['kmeans_model']
        self.is_fitted = model_data['is_fitted'] 
        self.n_clusters = model_data['n_clusters']
        
        print(f"[LOAD] Clustering model loaded from: {filepath}")
        print(f"[LOAD] Number of clusters: {model_data['n_clusters']}")
        print(f"[LOAD] Inertia: {model_data['inertia']:.4f}")


# =============================================================================
# EXAMPLE USAGE: TRAINING PHASE
# =============================================================================

def training_example():
    """
    Example of how to use these classes during training phase.
    This is what you would do in build_features.py
    """
    print("="*60)
    print("TRAINING PHASE EXAMPLE")
    print("="*60)
    
    # Create sample training data
    np.random.seed(42)
    training_data = pd.DataFrame({
        'acc_x_filled_lowpass': np.random.normal(0, 1, 1000),
        'acc_y_filled_lowpass': np.random.normal(0, 1, 1000),
        'acc_z_filled_lowpass': np.random.normal(9.8, 2, 1000),
        'gyr_x_filled_lowpass': np.random.normal(0, 10, 1000),
        'gyr_y_filled_lowpass': np.random.normal(0, 10, 1000),
        'gyr_z_filled_lowpass': np.random.normal(0, 10, 1000),
    })
    
    print(f"Training data shape: {training_data.shape}")
    
    # Define columns for PCA and clustering
    lowpass_cols = ['acc_x_filled_lowpass', 'acc_y_filled_lowpass', 'acc_z_filled_lowpass',
                    'gyr_x_filled_lowpass', 'gyr_y_filled_lowpass', 'gyr_z_filled_lowpass']
    cluster_cols = ['acc_x_filled_lowpass', 'acc_y_filled_lowpass', 'acc_z_filled_lowpass']
    
    # 1. APPLY PCA DURING TRAINING
    pca_transformer = ProductionReadyPCA()
    training_data = pca_transformer.fit_and_apply_pca(training_data, lowpass_cols, 3)
    
    # Save the PCA model for production use
    pca_transformer.save_pca_model("../../models/pca_model.pkl")
    
    # 2. APPLY CLUSTERING DURING TRAINING  
    clustering_transformer = ProductionReadyClustering()
    training_data = clustering_transformer.fit_and_apply_clustering(training_data, cluster_cols, 4)
    
    # Save the clustering model for production use
    clustering_transformer.save_clustering_model("../../models/clustering_model.pkl")
    
    print(f"\nFinal training data shape: {training_data.shape}")
    print(f"PCA columns: {[col for col in training_data.columns if 'pca' in col]}")
    print(f"Cluster column: {'cluster' in training_data.columns}")
    
    return training_data


# =============================================================================
# EXAMPLE USAGE: PREDICTION PHASE  
# =============================================================================

def prediction_example():
    """
    Example of how to use these classes during prediction phase.
    This is what you would do in predict_model.py
    """
    print("\n" + "="*60)
    print("PREDICTION PHASE EXAMPLE")
    print("="*60)
    
    # Create sample new data (simulating production input)
    np.random.seed(123)  # Different seed to simulate new data
    new_data = pd.DataFrame({
        'acc_x_filled_lowpass': np.random.normal(0.5, 1.2, 50),
        'acc_y_filled_lowpass': np.random.normal(-0.3, 0.8, 50),
        'acc_z_filled_lowpass': np.random.normal(10.2, 1.5, 50),
        'gyr_x_filled_lowpass': np.random.normal(2, 8, 50),
        'gyr_y_filled_lowpass': np.random.normal(-1, 12, 50),
        'gyr_z_filled_lowpass': np.random.normal(0.5, 15, 50),
    })
    
    print(f"New data shape: {new_data.shape}")
    
    # Define columns (must match training)
    lowpass_cols = ['acc_x_filled_lowpass', 'acc_y_filled_lowpass', 'acc_z_filled_lowpass',
                    'gyr_x_filled_lowpass', 'gyr_y_filled_lowpass', 'gyr_z_filled_lowpass']
    cluster_cols = ['acc_x_filled_lowpass', 'acc_y_filled_lowpass', 'acc_z_filled_lowpass']
    
    # 1. LOAD AND APPLY PCA FOR PREDICTION
    pca_transformer = ProductionReadyPCA()
    pca_transformer.load_pca_model("../../models/pca_model.pkl")
    new_data = pca_transformer.apply_pretrained_pca(new_data, lowpass_cols)
    
    # 2. LOAD AND APPLY CLUSTERING FOR PREDICTION
    clustering_transformer = ProductionReadyClustering()
    clustering_transformer.load_clustering_model("../../models/clustering_model.pkl") 
    new_data = clustering_transformer.apply_pretrained_clustering(new_data, cluster_cols)
    
    print(f"\nFinal prediction data shape: {new_data.shape}")
    print(f"PCA columns: {[col for col in new_data.columns if 'pca' in col]}")
    print(f"Cluster column: {'cluster' in new_data.columns}")
    
    return new_data


# =============================================================================
# INTEGRATION GUIDE FOR YOUR EXISTING CODE
# =============================================================================

def integration_guide():
    """
    Step-by-step guide for integrating into your existing codebase.
    """
    print("\n" + "="*60)
    print("INTEGRATION GUIDE FOR YOUR CODEBASE")
    print("="*60)
    
    integration_steps = """
    STEP 1: MODIFY YOUR EXISTING PCA CLASS (DataTransformation.py)
    --------------------------------------------------------------
    Replace the apply_pca method with:
    - fit_and_apply_pca() for training
    - apply_pretrained_pca() for prediction  
    - Add save_pca_model() and load_pca_model() methods
    
    STEP 2: UPDATE build_features.py (TRAINING)
    -------------------------------------------
    # Replace this line:
    df_pca = pca.apply_pca(df_pca, lowpass_cols, 3)
    
    # With this:
    df_pca = pca.fit_and_apply_pca(df_pca, lowpass_cols, 3)
    pca.save_pca_model("../../models/pca_model.pkl")  # You already added this!
    
    # For clustering, add after the clustering step:
    cluster_model.save_clustering_model("../../models/clustering_model.pkl")
    
    STEP 3: UPDATE predict_model.py (PREDICTION)
    --------------------------------------------
    In your ExercisePredictor.__init__():
    # Load pre-trained models
    self.pca.load_pca_model("../../models/pca_model.pkl")
    self.clustering.load_clustering_model("../../models/clustering_model.pkl")
    
    In apply_pca():
    # Replace:
    df_pca = self.pca.apply_pca(df_pca, lowpass_cols, 3)
    # With:
    df_pca = self.pca.apply_pretrained_pca(df_pca, lowpass_cols)
    
    In apply_clustering():  
    # Replace:
    df_cluster["cluster"] = kmeans.fit_predict(df_cluster[cluster_cols])
    # With:
    df_cluster = self.clustering.apply_pretrained_clustering(df_cluster, cluster_cols)
    
    STEP 4: KEY BENEFITS
    --------------------
    * Consistent PCA components between training and prediction
    * Consistent cluster assignments between training and prediction
    * Same normalization parameters used in both phases
    * No more "recomputing transformations" during prediction
    * Reproducible results with fixed random seeds
    
    STEP 5: TESTING
    ----------------
    1. Run training pipeline once to save models
    2. Run prediction pipeline multiple times on same data
    3. Verify you get identical results each time
    4. Compare feature values with original training features
    """
    
    print(integration_steps)


if __name__ == "__main__":
    # Run the complete example
    print("ML Model Persistence Guide - PCA and Clustering")
    
    # Show training phase
    training_data = training_example()
    
    # Show prediction phase
    prediction_data = prediction_example()
    
    # Show integration guide
    integration_guide()
    
    print("\n" + "="*60)
    print("GUIDE COMPLETED SUCCESSFULLY!")
    print("Next Steps:")
    print("   1. Update your DataTransformation.py with the new PCA class")
    print("   2. Add clustering model saving to build_features.py")  
    print("   3. Update predict_model.py to load and use pre-trained models")
    print("   4. Test the pipeline end-to-end")
    print("="*60)