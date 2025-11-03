"""
Unit tests for feature extraction module.

Tests cover:
- Feature extraction with TSFresh
- DataFrame preparation
- FITS file processing
- Batch extraction
- Feature set configurations
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.detection.feature_extractor import FeatureExtractor, extract_features_batch


@pytest.fixture
def feature_extractor():
    """Create a feature extractor for testing."""
    return FeatureExtractor(
        feature_set='minimal',
        n_jobs=1,
        disable_progressbar=True
    )


@pytest.fixture
def sample_light_curves():
    """Create sample light curves for testing."""
    n_lcs = 3
    n_points = 100
    
    light_curves = []
    for i in range(n_lcs):
        lc = {
            'time': np.linspace(0, 10, n_points),
            'flux': np.ones(n_points) + 0.01 * np.random.randn(n_points)
        }
        light_curves.append(lc)
    
    return light_curves


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""
    
    def test_initialization(self, feature_extractor):
        """Test feature extractor initialization."""
        assert feature_extractor.feature_set == 'minimal'
        assert feature_extractor.n_jobs == 1
        assert feature_extractor.disable_progressbar is True
    
    def test_initialization_invalid_feature_set(self):
        """Test initialization with invalid feature set."""
        with pytest.raises(ValueError, match="Unknown feature_set"):
            FeatureExtractor(feature_set='invalid')
    
    def test_prepare_dataframe(self, feature_extractor, sample_light_curves):
        """Test DataFrame preparation for TSFresh."""
        ids = ['lc_0', 'lc_1', 'lc_2']
        df = feature_extractor.prepare_dataframe(sample_light_curves, ids)
        
        # Verify DataFrame structure
        assert isinstance(df, pd.DataFrame)
        assert 'id' in df.columns
        assert 'time' in df.columns
        assert 'flux' in df.columns
        assert len(df['id'].unique()) == 3
        assert len(df) == 3 * 100  # 3 LCs * 100 points each
    
    def test_prepare_dataframe_auto_ids(self, feature_extractor, sample_light_curves):
        """Test DataFrame preparation with automatic ID generation."""
        df = feature_extractor.prepare_dataframe(sample_light_curves)
        
        # Should generate IDs automatically
        assert len(df['id'].unique()) == 3
        assert all('lc_' in str(id_) for id_ in df['id'].unique())
    
    def test_prepare_dataframe_length_mismatch(self, feature_extractor, sample_light_curves):
        """Test error handling for ID/LC length mismatch."""
        with pytest.raises(ValueError, match="Length mismatch"):
            feature_extractor.prepare_dataframe(sample_light_curves, ids=['lc_0'])
    
    @patch('src.detection.feature_extractor.extract_features')
    def test_extract(self, mock_extract_features, feature_extractor, sample_light_curves):
        """Test feature extraction."""
        # Mock TSFresh extraction
        mock_features = pd.DataFrame({
            'feature_1': [1.0, 2.0, 3.0],
            'feature_2': [4.0, 5.0, 6.0]
        }, index=['lc_0', 'lc_1', 'lc_2'])
        mock_extract_features.return_value = mock_features
        
        features = feature_extractor.extract(sample_light_curves)
        
        # Verify output
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 3
        assert 'feature_1' in features.columns
        mock_extract_features.assert_called_once()
    
    @patch('src.detection.feature_extractor.FeatureExtractor.extract')
    @patch('src.detection.feature_extractor.LightCurvePreprocessor.preprocess')
    def test_extract_from_fits(self, mock_preprocess, mock_extract, feature_extractor):
        """Test extraction from FITS files."""
        # Mock preprocessing
        mock_lc = {
            'time': np.linspace(0, 10, 100),
            'flux': np.ones(100)
        }
        mock_preprocess.return_value = (mock_lc, {'n_points': 100})
        
        # Mock feature extraction
        mock_features = pd.DataFrame({
            'feature_1': [1.0, 2.0]
        }, index=['test_0', 'test_1'])
        mock_extract.return_value = mock_features
        
        fits_files = [Path('test_0.fits'), Path('test_1.fits')]
        labels = [1, 0]
        
        features, valid_ids, valid_labels = feature_extractor.extract_from_fits(
            fits_files,
            labels=labels
        )
        
        # Verify output
        assert isinstance(features, pd.DataFrame)
        assert len(valid_ids) == 2
        assert valid_labels == [1, 0]
    
    @patch('src.detection.feature_extractor.FeatureExtractor.extract_from_fits')
    def test_extract_and_save(self, mock_extract_from_fits, feature_extractor, tmp_path):
        """Test extraction and saving to CSV."""
        # Mock extraction
        mock_features = pd.DataFrame({
            'feature_1': [1.0, 2.0],
            'feature_2': [3.0, 4.0]
        })
        mock_extract_from_fits.return_value = (mock_features, ['lc_0', 'lc_1'], [1, 0])
        
        output_path = tmp_path / "features.csv"
        fits_files = [Path('test_0.fits'), Path('test_1.fits')]
        
        stats = feature_extractor.extract_and_save(
            fits_files,
            output_path,
            labels=[1, 0]
        )
        
        # Verify output
        assert stats['n_samples'] == 2
        assert stats['n_features'] == 2
        assert stats['feature_set'] == 'minimal'


class TestFeatureSetConfigurations:
    """Test different feature set configurations."""
    
    def test_comprehensive_feature_set(self):
        """Test comprehensive feature set initialization."""
        extractor = FeatureExtractor(feature_set='comprehensive', n_jobs=1)
        assert extractor.feature_set == 'comprehensive'
    
    def test_efficient_feature_set(self):
        """Test efficient feature set initialization."""
        extractor = FeatureExtractor(feature_set='efficient', n_jobs=1)
        assert extractor.feature_set == 'efficient'
    
    def test_minimal_feature_set(self):
        """Test minimal feature set initialization."""
        extractor = FeatureExtractor(feature_set='minimal', n_jobs=1)
        assert extractor.feature_set == 'minimal'
        assert 'mean' in extractor.FEATURE_SETS['minimal']


class TestBatchExtraction:
    """Test batch extraction functionality."""
    
    @patch('src.detection.feature_extractor.FeatureExtractor.extract_and_save')
    @patch('pathlib.Path.glob')
    @patch('pandas.read_csv')
    def test_extract_features_batch(self, mock_read_csv, mock_glob, mock_extract_and_save, tmp_path):
        """Test batch feature extraction."""
        # Mock file finding
        mock_glob.return_value = [Path('test_0.fits'), Path('test_1.fits')]
        
        # Mock labels
        mock_labels_df = pd.DataFrame({
            'id': ['test_0', 'test_1'],
            'label': [1, 0]
        })
        mock_read_csv.return_value = mock_labels_df
        
        # Mock extraction
        mock_extract_and_save.return_value = {
            'n_samples': 2,
            'n_features': 10
        }
        
        output_path = tmp_path / "features.csv"
        stats = extract_features_batch(
            input_dir=tmp_path,
            output_path=output_path,
            labels_file=tmp_path / "labels.csv",
            feature_set='minimal',
            n_jobs=1
        )
        
        # Verify
        assert stats['n_samples'] == 2
        assert stats['n_features'] == 10