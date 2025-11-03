"""
Unit tests for light curve preprocessing module.

Tests cover:
- FITS file reading
- NaN removal
- Outlier detection
- Normalization
- Detrending
- Quality metrics
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from astropy.io import fits
from astropy.table import Table

from src.detection.preprocess import LightCurvePreprocessor, preprocess_light_curve_batch


@pytest.fixture
def preprocessor():
    """Create a preprocessor instance for testing."""
    return LightCurvePreprocessor(
        normalize_method="median",
        outlier_sigma=5.0,
        min_valid_points=50
    )


@pytest.fixture
def sample_light_curve():
    """Create a sample light curve for testing."""
    n_points = 1000
    time = np.linspace(0, 100, n_points)
    flux = np.ones(n_points) + 0.01 * np.random.randn(n_points)
    flux_err = 0.001 * np.ones(n_points)
    
    return {
        'time': time,
        'flux': flux,
        'flux_err': flux_err
    }


@pytest.fixture
def sample_fits_data():
    """Create mock FITS data."""
    n_points = 1000
    time = np.linspace(0, 100, n_points)
    flux = np.ones(n_points) + 0.01 * np.random.randn(n_points)
    flux_err = 0.001 * np.ones(n_points)
    
    return {
        'TIME': time,
        'PDCSAP_FLUX': flux,
        'PDCSAP_FLUX_ERR': flux_err
    }


class TestLightCurvePreprocessor:
    """Test suite for LightCurvePreprocessor class."""
    
    def test_initialization(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.normalize_method == "median"
        assert preprocessor.outlier_sigma == 5.0
        assert preprocessor.min_valid_points == 50
    
    @patch('astropy.io.fits.open')
    def test_read_fits_kepler(self, mock_fits_open, preprocessor, sample_fits_data):
        """Test reading Kepler FITS files."""
        # Mock FITS file
        mock_hdul = MagicMock()
        mock_data = Table(sample_fits_data)
        mock_hdul.__enter__.return_value = [None, Mock(data=mock_data)]
        mock_hdul.__len__.return_value = 2
        mock_fits_open.return_value = mock_hdul
        
        # Read FITS file
        lc = preprocessor.read_fits("dummy.fits")
        
        # Verify output
        assert 'time' in lc
        assert 'flux' in lc
        assert 'flux_err' in lc
        assert len(lc['time']) == len(sample_fits_data['TIME'])
        assert lc['time'].dtype == np.float64
        assert lc['flux'].dtype == np.float32
    
    @patch('astropy.io.fits.open')
    def test_read_fits_tess(self, mock_fits_open, preprocessor):
        """Test reading TESS FITS files."""
        # Mock FITS data with TESS column names
        n_points = 100
        tess_data = {
            'TIME': np.linspace(0, 10, n_points),
            'SAP_FLUX': np.ones(n_points),
            'SAP_FLUX_ERR': 0.001 * np.ones(n_points)
        }
        
        mock_hdul = MagicMock()
        mock_data = Table(tess_data)
        mock_hdul.__enter__.return_value = [None, Mock(data=mock_data)]
        mock_hdul.__len__.return_value = 2
        mock_fits_open.return_value = mock_hdul
        
        lc = preprocessor.read_fits("dummy_tess.fits")
        
        assert len(lc['flux']) == n_points
    
    def test_remove_nans(self, preprocessor, sample_light_curve):
        """Test NaN removal."""
        # Add some NaNs
        lc = sample_light_curve.copy()
        lc['flux'][10:15] = np.nan
        lc['time'][20:25] = np.nan
        
        # Remove NaNs
        cleaned = preprocessor.remove_nans(lc)
        
        # Verify no NaNs remain
        assert not np.any(np.isnan(cleaned['time']))
        assert not np.any(np.isnan(cleaned['flux']))
        assert not np.any(np.isnan(cleaned['flux_err']))
        assert len(cleaned['flux']) < len(lc['flux'])
    
    def test_remove_outliers(self, preprocessor, sample_light_curve):
        """Test outlier removal."""
        # Add outliers
        lc = sample_light_curve.copy()
        lc['flux'][50] = 10.0  # Strong outlier
        lc['flux'][100] = -5.0  # Strong outlier
        
        # Remove outliers
        cleaned = preprocessor.remove_outliers(lc)
        
        # Verify outliers removed
        assert len(cleaned['flux']) < len(lc['flux'])
        assert np.max(cleaned['flux']) < 10.0
        assert np.min(cleaned['flux']) > -5.0
    
    def test_normalize_flux_median(self, preprocessor, sample_light_curve):
        """Test median normalization."""
        lc = sample_light_curve.copy()
        normalized = preprocessor.normalize_flux(lc)
        
        # Verify normalization
        assert np.median(normalized['flux']) == pytest.approx(1.0, rel=0.1)
        assert normalized['flux'].dtype == np.float32
    
    def test_normalize_flux_mean(self, sample_light_curve):
        """Test mean normalization."""
        preprocessor = LightCurvePreprocessor(normalize_method="mean")
        lc = sample_light_curve.copy()
        normalized = preprocessor.normalize_flux(lc)
        
        assert np.mean(normalized['flux']) == pytest.approx(1.0, rel=0.1)
    
    def test_normalize_flux_minmax(self, sample_light_curve):
        """Test min-max normalization."""
        preprocessor = LightCurvePreprocessor(normalize_method="minmax")
        lc = sample_light_curve.copy()
        normalized = preprocessor.normalize_flux(lc)
        
        assert np.min(normalized['flux']) >= 0.0
        assert np.max(normalized['flux']) <= 1.0
    
    def test_detrend_flux(self, preprocessor, sample_light_curve):
        """Test flux detrending."""
        # Add a linear trend
        lc = sample_light_curve.copy()
        lc['flux'] = lc['flux'] + 0.001 * lc['time']
        
        # Detrend
        detrended = preprocessor.detrend_flux(lc)
        
        # Verify trend is removed (flux should be more stationary)
        original_std = np.std(lc['flux'])
        detrended_std = np.std(detrended['flux'])
        assert detrended_std < original_std
    
    def test_calculate_quality_metrics(self, preprocessor, sample_light_curve):
        """Test quality metrics calculation."""
        lc = sample_light_curve.copy()
        metrics = preprocessor.calculate_quality_metrics(lc)
        
        # Verify metrics
        assert 'n_points' in metrics
        assert 'time_span_days' in metrics
        assert 'mean_flux' in metrics
        assert 'std_flux' in metrics
        assert 'snr' in metrics
        assert metrics['n_points'] == len(lc['flux'])
        assert metrics['snr'] > 0
    
    @patch('src.detection.preprocess.LightCurvePreprocessor.read_fits')
    def test_preprocess_pipeline(self, mock_read_fits, preprocessor, sample_light_curve):
        """Test complete preprocessing pipeline."""
        mock_read_fits.return_value = sample_light_curve
        
        lc, metrics = preprocessor.preprocess("dummy.fits", detrend=True)
        
        # Verify output
        assert 'time' in lc
        assert 'flux' in lc
        assert 'flux_err' in lc
        assert 'n_points' in metrics
        assert len(lc['flux']) >= preprocessor.min_valid_points


class TestBatchPreprocessing:
    """Test batch preprocessing functionality."""
    
    @patch('src.detection.preprocess.LightCurvePreprocessor.preprocess')
    def test_preprocess_light_curve_batch(self, mock_preprocess, sample_light_curve):
        """Test batch preprocessing."""
        # Mock preprocessing
        mock_preprocess.return_value = (sample_light_curve, {'n_points': len(sample_light_curve['flux'])})
        
        fits_files = [Path(f"test_{i}.fits") for i in range(5)]
        results = preprocess_light_curve_batch(fits_files)
        
        # Verify results
        assert len(results) == 5
        assert all(len(r) == 3 for r in results)  # (path, lc, metrics)
    
    @patch('src.detection.preprocess.LightCurvePreprocessor.preprocess')
    def test_batch_with_failures(self, mock_preprocess, sample_light_curve):
        """Test batch preprocessing with some failures."""
        # Mock some failures
        def side_effect(fits_path):
            if "fail" in str(fits_path):
                raise ValueError("Mock failure")
            return (sample_light_curve, {'n_points': len(sample_light_curve['flux'])})
        
        mock_preprocess.side_effect = side_effect
        
        fits_files = [Path(f"test_{i}.fits") for i in range(3)] + [Path("fail.fits")]
        results = preprocess_light_curve_batch(fits_files)
        
        # Should have 3 successful results
        assert len(results) == 3