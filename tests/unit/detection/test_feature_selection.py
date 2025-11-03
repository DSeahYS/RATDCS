"""
Unit tests for feature selection module.

Tests cover:
- TSFresh relevance-based selection
- Mutual information selection
- F-test selection
- Random forest importance
- Correlation-based filtering
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch

from src.detection.feature_selection import FeatureSelector, calculate_feature_importance


@pytest.fixture
def sample_features():
    """Create sample features for testing."""
    np.random.seed(42)
    n_samples = 100
    n_features = 20
    
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    return features


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    np.random.seed(42)
    return pd.Series(np.random.randint(0, 2, 100))


class TestFeatureSelector:
    """Test suite for FeatureSelector class."""
    
    def test_initialization(self):
        """Test feature selector initialization."""
        selector = FeatureSelector(
            method='mutual_info',
            k_features=10,
            correlation_threshold=0.9
        )
        
        assert selector.method == 'mutual_info'
        assert selector.k_features == 10
        assert selector.correlation_threshold == 0.9
    
    def test_initialization_invalid_method(self):
        """Test initialization with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            FeatureSelector(method='invalid')
    
    def test_select_features_mutual_info(self, sample_features, sample_labels):
        """Test mutual information-based selection."""
        selector = FeatureSelector(method='mutual_info', k_features=10)
        
        selected, scores = selector.select_features_mutual_info(
            sample_features,
            sample_labels,
            k=10
        )
        
        assert len(selected) == 10
        assert len(scores) == len(sample_features.columns)
        assert all(feat in sample_features.columns for feat in selected)
    
    def test_select_features_f_test(self, sample_features, sample_labels):
        """Test F-test based selection."""
        selector = FeatureSelector(method='f_test', k_features=10)
        
        selected, (f_scores, p_values) = selector.select_features_f_test(
            sample_features,
            sample_labels,
            k=10
        )
        
        assert len(selected) == 10
        assert len(f_scores) == len(sample_features.columns)
        assert len(p_values) == len(sample_features.columns)
    
    def test_select_features_random_forest(self, sample_features, sample_labels):
        """Test random forest importance-based selection."""
        selector = FeatureSelector(method='random_forest', k_features=10)
        
        selected, importances = selector.select_features_random_forest(
            sample_features,
            sample_labels,
            k=10
        )
        
        assert len(selected) == 10
        assert len(importances) == len(sample_features.columns)
        assert all(imp >= 0 for imp in importances)
    
    def test_remove_correlated_features(self, sample_features):
        """Test correlation-based feature removal."""
        # Add highly correlated features
        features = sample_features.copy()
        features['correlated_1'] = features['feature_0'] + 0.01 * np.random.randn(len(features))
        features['correlated_2'] = features['feature_1'] + 0.01 * np.random.randn(len(features))
        
        selector = FeatureSelector(correlation_threshold=0.95)
        to_keep = selector.remove_correlated_features(features)
        
        # Should remove some features
        assert len(to_keep) < len(features.columns)
    
    def test_select_mutual_info_method(self, sample_features, sample_labels):
        """Test complete selection pipeline with mutual info."""
        selector = FeatureSelector(method='mutual_info', k_features=10)
        
        selected, info = selector.select(
            sample_features,
            sample_labels,
            remove_correlated=False
        )
        
        assert len(selected) == 10
        assert info['method'] == 'mutual_info'
        assert 'mi_scores' in info
        assert info['n_selected'] == 10
    
    @patch('pandas.read_csv')
    def test_select_and_save(self, mock_read_csv, tmp_path, sample_features, sample_labels):
        """Test selection and saving to file."""
        # Mock CSV reading
        df = sample_features.copy()
        df['label'] = sample_labels
        df['id'] = [f'lc_{i}' for i in range(len(df))]
        mock_read_csv.return_value = df
        
        selector = FeatureSelector(method='mutual_info', k_features=10)
        
        input_path = tmp_path / "features.csv"
        output_path = tmp_path / "selected_features.csv"
        
        stats = selector.select_and_save(
            features_path=input_path,
            output_path=output_path
        )
        
        assert stats['n_features_input'] == len(sample_features.columns)
        assert stats['n_features_selected'] <= len(sample_features.columns)


class TestFeatureImportance:
    """Test feature importance calculation."""
    
    def test_calculate_feature_importance(self, sample_features, sample_labels):
        """Test multi-method importance calculation."""
        importance_df = calculate_feature_importance(
            sample_features,
            sample_labels,
            top_k=10
        )
        
        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) <= 10
        assert 'feature' in importance_df.columns
        assert 'mutual_info' in importance_df.columns
        assert 'f_score' in importance_df.columns
        assert 'rf_importance' in importance_df.columns
        assert 'avg_rank' in importance_df.columns