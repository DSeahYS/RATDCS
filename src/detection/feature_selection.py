"""
Feature Selection for Exoplanet Classification

This module provides feature selection and importance analysis for time-series features
extracted from light curves using TSFresh.

Based on RATDCS ARCHITECTURE.md Section 3.1.2 - Exoplanet Candidate Classifier
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from tsfresh.feature_selection.relevance import calculate_relevance_table

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Select relevant features and calculate feature importance.
    
    Provides multiple feature selection methods including:
    - TSFresh relevance-based selection
    - Mutual information
    - ANOVA F-test
    - Random forest importance
    - Correlation-based filtering
    
    Attributes:
        method (str): Selection method to use
        k_features (int): Number of features to select (None = auto)
        correlation_threshold (float): Threshold for removing correlated features
    """
    
    SELECTION_METHODS = ['tsfresh', 'mutual_info', 'f_test', 'random_forest', 'all']
    
    def __init__(
        self,
        method: str = 'tsfresh',
        k_features: Optional[int] = None,
        correlation_threshold: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize the feature selector.
        
        Args:
            method: Selection method ('tsfresh', 'mutual_info', 'f_test', 'random_forest', 'all')
            k_features: Number of features to select (None = auto-determine)
            correlation_threshold: Remove features with correlation above this threshold
            random_state: Random state for reproducibility
        """
        if method not in self.SELECTION_METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from: {self.SELECTION_METHODS}")
        
        self.method = method
        self.k_features = k_features
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state
        
        logger.info(f"Initialized FeatureSelector with method='{method}'")
    
    def select_features_tsfresh(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        fdr_level: float = 0.05
    ) -> Tuple[List[str], pd.DataFrame]:
        """
        Select features using TSFresh's relevance-based selection.
        
        Args:
            features: Feature dataframe
            labels: Target labels
            fdr_level: False discovery rate level (default: 0.05)
        
        Returns:
            Tuple of (selected_feature_names, relevance_table)
        """
        logger.info("Selecting features using TSFresh relevance test...")
        
        # Calculate relevance
        relevance_table = calculate_relevance_table(
            features,
            labels,
            fdr_level=fdr_level,
            n_jobs=4
        )
        
        # Select relevant features
        selected_features = relevance_table[relevance_table['relevant']]['feature'].tolist()
        
        logger.info(f"Selected {len(selected_features)}/{len(features.columns)} features")
        
        return selected_features, relevance_table
    
    def select_features_mutual_info(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        k: Optional[int] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select features using mutual information.
        
        Args:
            features: Feature dataframe
            labels: Target labels
            k: Number of features to select (default: self.k_features or 'all')
        
        Returns:
            Tuple of (selected_feature_names, mutual_info_scores)
        """
        logger.info("Selecting features using mutual information...")
        
        k = k or self.k_features or 'all'
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(
            features,
            labels,
            discrete_features=False,
            random_state=self.random_state
        )
        
        # Select k best features
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(features, labels)
        
        selected_features = features.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)}/{len(features.columns)} features")
        
        return selected_features, mi_scores
    
    def select_features_f_test(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        k: Optional[int] = None
    ) -> Tuple[List[str], Tuple[np.ndarray, np.ndarray]]:
        """
        Select features using ANOVA F-test.
        
        Args:
            features: Feature dataframe
            labels: Target labels
            k: Number of features to select (default: self.k_features or 'all')
        
        Returns:
            Tuple of (selected_feature_names, (f_scores, p_values))
        """
        logger.info("Selecting features using ANOVA F-test...")
        
        k = k or self.k_features or 'all'
        
        # Calculate F-test scores
        f_scores, p_values = f_classif(features, labels)
        
        # Select k best features
        selector = SelectKBest(f_classif, k=k)
        selector.fit(features, labels)
        
        selected_features = features.columns[selector.get_support()].tolist()
        
        logger.info(f"Selected {len(selected_features)}/{len(features.columns)} features")
        
        return selected_features, (f_scores, p_values)
    
    def select_features_random_forest(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        k: Optional[int] = None,
        n_estimators: int = 100
    ) -> Tuple[List[str], np.ndarray]:
        """
        Select features using random forest feature importance.
        
        Args:
            features: Feature dataframe
            labels: Target labels
            k: Number of features to select (default: self.k_features or all with importance > 0)
            n_estimators: Number of trees in random forest
        
        Returns:
            Tuple of (selected_feature_names, feature_importances)
        """
        logger.info("Selecting features using Random Forest importance...")
        
        # Train random forest
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=4
        )
        rf.fit(features, labels)
        
        # Get feature importances
        importances = rf.feature_importances_
        
        # Select k best features
        k = k or self.k_features
        if k:
            indices = np.argsort(importances)[-k:]
            selected_features = features.columns[indices].tolist()
        else:
            # Select all features with non-zero importance
            selected_features = features.columns[importances > 0].tolist()
        
        logger.info(f"Selected {len(selected_features)}/{len(features.columns)} features")
        
        return selected_features, importances
    
    def remove_correlated_features(
        self,
        features: pd.DataFrame,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Remove highly correlated features.
        
        Args:
            features: Feature dataframe
            threshold: Correlation threshold (default: self.correlation_threshold)
        
        Returns:
            List of features to keep
        """
        threshold = threshold or self.correlation_threshold
        
        logger.info(f"Removing features with correlation > {threshold}...")
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find features to remove
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_remove = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]
        
        # Keep remaining features
        to_keep = [col for col in features.columns if col not in to_remove]
        
        logger.info(f"Removed {len(to_remove)} correlated features, keeping {len(to_keep)}")
        
        return to_keep
    
    def select(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        remove_correlated: bool = True
    ) -> Tuple[List[str], Dict]:
        """
        Select features using the configured method.
        
        Args:
            features: Feature dataframe
            labels: Target labels
            remove_correlated: Whether to remove highly correlated features
        
        Returns:
            Tuple of (selected_feature_names, selection_info_dict)
        """
        # Remove correlated features first if requested
        if remove_correlated:
            features_clean = features[self.remove_correlated_features(features)]
        else:
            features_clean = features
        
        # Select features using specified method
        selection_info = {'method': self.method}
        
        if self.method == 'tsfresh':
            selected, relevance = self.select_features_tsfresh(features_clean, labels)
            selection_info['relevance_table'] = relevance
            
        elif self.method == 'mutual_info':
            selected, scores = self.select_features_mutual_info(features_clean, labels)
            selection_info['mi_scores'] = dict(zip(features_clean.columns, scores))
            
        elif self.method == 'f_test':
            selected, (f_scores, p_values) = self.select_features_f_test(features_clean, labels)
            selection_info['f_scores'] = dict(zip(features_clean.columns, f_scores))
            selection_info['p_values'] = dict(zip(features_clean.columns, p_values))
            
        elif self.method == 'random_forest':
            selected, importances = self.select_features_random_forest(features_clean, labels)
            selection_info['importances'] = dict(zip(features_clean.columns, importances))
            
        elif self.method == 'all':
            # Use all selection methods and take intersection
            selected_tsfresh, _ = self.select_features_tsfresh(features_clean, labels)
            selected_mi, _ = self.select_features_mutual_info(features_clean, labels)
            selected_rf, _ = self.select_features_random_forest(features_clean, labels)
            
            # Intersection of all methods
            selected = list(set(selected_tsfresh) & set(selected_mi) & set(selected_rf))
            selection_info['methods_used'] = ['tsfresh', 'mutual_info', 'random_forest']
            selection_info['n_selected_per_method'] = {
                'tsfresh': len(selected_tsfresh),
                'mutual_info': len(selected_mi),
                'random_forest': len(selected_rf),
                'intersection': len(selected)
            }
        
        selection_info['n_selected'] = len(selected)
        selection_info['n_total'] = len(features.columns)
        selection_info['selected_features'] = selected
        
        return selected, selection_info
    
    def select_and_save(
        self,
        features_path: Union[str, Path],
        output_path: Union[str, Path],
        label_column: str = 'label',
        id_column: str = 'id',
        remove_correlated: bool = True
    ) -> Dict:
        """
        Select features from CSV file and save results.
        
        Args:
            features_path: Path to input CSV with features
            output_path: Path to output CSV with selected features
            label_column: Name of label column
            id_column: Name of ID column
            remove_correlated: Whether to remove correlated features
        
        Returns:
            Dictionary with selection statistics
        """
        # Load features
        df = pd.read_csv(features_path)
        
        if label_column not in df.columns:
            raise ValueError(f"Label column '{label_column}' not found in {features_path}")
        
        # Separate features and labels
        exclude_cols = [label_column]
        if id_column in df.columns:
            exclude_cols.append(id_column)
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        features = df[feature_cols]
        labels = df[label_column]
        
        # Select features
        selected_features, selection_info = self.select(
            features,
            labels,
            remove_correlated=remove_correlated
        )
        
        # Create output dataframe
        output_cols = exclude_cols + selected_features
        output_df = df[output_cols]
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        
        logger.info(f"Saved selected features to {output_path}")
        
        # Save selection info
        info_path = output_path.with_suffix('.json')
        
        # Convert numpy types to Python types for JSON serialization
        selection_info_json = {}
        for key, value in selection_info.items():
            if isinstance(value, (np.ndarray, pd.Series)):
                selection_info_json[key] = value.tolist()
            elif isinstance(value, dict):
                selection_info_json[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in value.items()
                }
            else:
                selection_info_json[key] = value
        
        with open(info_path, 'w') as f:
            json.dump(selection_info_json, f, indent=2)
        
        logger.info(f"Saved selection info to {info_path}")
        
        return {
            'input_file': str(features_path),
            'output_file': str(output_path),
            'n_samples': len(output_df),
            'n_features_input': len(feature_cols),
            'n_features_selected': len(selected_features),
            'selection_method': self.method,
            'selection_ratio': len(selected_features) / len(feature_cols)
        }


def calculate_feature_importance(
    features: pd.DataFrame,
    labels: pd.Series,
    top_k: int = 50
) -> pd.DataFrame:
    """
    Calculate and rank feature importance using multiple methods.
    
    Args:
        features: Feature dataframe
        labels: Target labels
        top_k: Number of top features to return
    
    Returns:
        DataFrame with feature importance scores from multiple methods
    """
    importance_df = pd.DataFrame({'feature': features.columns})
    
    # Mutual Information
    logger.info("Calculating mutual information...")
    mi_scores = mutual_info_classif(features, labels, random_state=42)
    importance_df['mutual_info'] = mi_scores
    
    # F-test
    logger.info("Calculating F-test scores...")
    f_scores, p_values = f_classif(features, labels)
    importance_df['f_score'] = f_scores
    importance_df['p_value'] = p_values
    
    # Random Forest
    logger.info("Training Random Forest for importance...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=4)
    rf.fit(features, labels)
    importance_df['rf_importance'] = rf.feature_importances_
    
    # Calculate average rank across methods
    for col in ['mutual_info', 'f_score', 'rf_importance']:
        importance_df[f'{col}_rank'] = importance_df[col].rank(ascending=False)
    
    importance_df['avg_rank'] = importance_df[
        ['mutual_info_rank', 'f_score_rank', 'rf_importance_rank']
    ].mean(axis=1)
    
    # Sort by average rank
    importance_df = importance_df.sort_values('avg_rank')
    
    return importance_df.head(top_k)