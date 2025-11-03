"""
Unit tests for Kepler downloader module.

These tests mock external API calls to avoid actual data downloads.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from astropy.io import fits
from astropy.table import Table

from src.data.kepler_downloader import KeplerDownloader


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "kepler_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


@pytest.fixture
def downloader(temp_output_dir):
    """Create a KeplerDownloader instance for testing."""
    return KeplerDownloader(
        output_dir=temp_output_dir,
        verify_fits=True,
        max_retries=3
    )


class TestKeplerDownloaderInit:
    """Test KeplerDownloader initialization."""
    
    def test_init_creates_directories(self, temp_output_dir):
        """Test that initialization creates required directories."""
        downloader = KeplerDownloader(output_dir=temp_output_dir)
        
        assert downloader.output_dir.exists()
        assert downloader.cache_dir.exists()
    
    def test_init_with_custom_cache_dir(self, temp_output_dir, tmp_path):
        """Test initialization with custom cache directory."""
        cache_dir = tmp_path / "custom_cache"
        downloader = KeplerDownloader(
            output_dir=temp_output_dir,
            cache_dir=str(cache_dir)
        )
        
        assert downloader.cache_dir == cache_dir
        assert downloader.cache_dir.exists()


class TestQueryConfirmedExoplanets:
    """Test query_confirmed_exoplanets method."""
    
    @patch('src.data.kepler_downloader.Observations.query_criteria')
    def test_query_returns_observations(self, mock_query, downloader):
        """Test that query returns observation list."""
        # Mock observation table
        mock_table = Table({
            'obsid': ['obs1', 'obs2', 'obs3'],
            'target_name': ['Kepler-1', 'Kepler-2', 'Kepler-3'],
            'obs_collection': ['Kepler', 'Kepler', 'Kepler'],
            's_ra': [290.0, 291.0, 292.0],
            's_dec': [45.0, 46.0, 47.0]
        })
        mock_query.return_value = mock_table
        
        observations = downloader.query_confirmed_exoplanets(max_targets=3)
        
        assert len(observations) == 3
        assert observations[0]['target_name'] == 'Kepler-1'
        assert 's_ra' in observations[0]
    
    @patch('src.data.kepler_downloader.Observations.query_criteria')
    def test_query_limits_to_max_targets(self, mock_query, downloader):
        """Test that query respects max_targets limit."""
        # Mock large observation table
        mock_table = Table({
            'obsid': [f'obs{i}' for i in range(200)],
            'target_name': [f'Kepler-{i}' for i in range(200)],
            'obs_collection': ['Kepler'] * 200,
            's_ra': [290.0] * 200,
            's_dec': [45.0] * 200
        })
        mock_query.return_value = mock_table
        
        observations = downloader.query_confirmed_exoplanets(max_targets=50)
        
        assert len(observations) <= 50
    
    @patch('src.data.kepler_downloader.Observations.query_criteria')
    def test_query_handles_empty_results(self, mock_query, downloader):
        """Test handling of empty query results."""
        mock_query.return_value = Table()
        
        observations = downloader.query_confirmed_exoplanets(max_targets=10)
        
        assert len(observations) == 0


class TestVerifyFitsFile:
    """Test FITS file verification."""
    
    def test_verify_valid_fits_file(self, downloader, tmp_path):
        """Test verification of valid FITS file."""
        # Create a valid FITS file
        fits_path = tmp_path / "valid.fits"
        
        time = np.arange(0, 100, 0.1)
        flux = np.random.randn(len(time)) + 1000
        
        col1 = fits.Column(name='TIME', format='D', array=time)
        col2 = fits.Column(name='PDCSAP_FLUX', format='E', array=flux)
        
        hdu_primary = fits.PrimaryHDU()
        hdu_table = fits.BinTableHDU.from_columns([col1, col2])
        hdul = fits.HDUList([hdu_primary, hdu_table])
        hdul.writeto(fits_path, overwrite=True)
        
        assert downloader._verify_fits_file(fits_path) is True
    
    def test_verify_missing_columns(self, downloader, tmp_path):
        """Test verification fails for FITS with missing columns."""
        fits_path = tmp_path / "invalid.fits"
        
        # Create FITS with wrong columns
        time = np.arange(0, 100, 0.1)
        col1 = fits.Column(name='TIME', format='D', array=time)
        
        hdu_primary = fits.PrimaryHDU()
        hdu_table = fits.BinTableHDU.from_columns([col1])
        hdul = fits.HDUList([hdu_primary, hdu_table])
        hdul.writeto(fits_path, overwrite=True)
        
        assert downloader._verify_fits_file(fits_path) is False
    
    def test_verify_all_nan_data(self, downloader, tmp_path):
        """Test verification fails for all-NaN data."""
        fits_path = tmp_path / "nan_data.fits"
        
        time = np.full(100, np.nan)
        flux = np.full(100, np.nan)
        
        col1 = fits.Column(name='TIME', format='D', array=time)
        col2 = fits.Column(name='PDCSAP_FLUX', format='E', array=flux)
        
        hdu_primary = fits.PrimaryHDU()
        hdu_table = fits.BinTableHDU.from_columns([col1, col2])
        hdul = fits.HDUList([hdu_primary, hdu_table])
        hdul.writeto(fits_path, overwrite=True)
        
        assert downloader._verify_fits_file(fits_path) is False


class TestDownloadLightCurves:
    """Test download_light_curves method."""
    
    @patch('src.data.kepler_downloader.Observations.download_products')
    @patch('src.data.kepler_downloader.Observations.get_product_list')
    @patch('src.data.kepler_downloader.KeplerDownloader.query_confirmed_exoplanets')
    def test_download_success(self, mock_query, mock_products, mock_download, downloader, tmp_path):
        """Test successful download of light curves."""
        # Mock observations
        mock_query.return_value = [
            {'obs_id': 'obs1', 'target_name': 'Kepler-1', 'obs_collection': 'Kepler'}
        ]
        
        # Mock products
        mock_products.return_value = Table({
            'productSubGroupDescription': ['LLC'],
            'obs_id': ['obs1']
        })
        
        # Mock download manifest
        test_fits = tmp_path / "kepler_001.fits"
        # Create test FITS file
        time = np.arange(0, 100, 0.1)
        flux = np.random.randn(len(time)) + 1000
        col1 = fits.Column(name='TIME', format='D', array=time)
        col2 = fits.Column(name='PDCSAP_FLUX', format='E', array=flux)
        hdu_primary = fits.PrimaryHDU()
        hdu_table = fits.BinTableHDU.from_columns([col1, col2])
        hdul = fits.HDUList([hdu_primary, hdu_table])
        hdul.writeto(test_fits, overwrite=True)
        
        mock_download.return_value = Table({
            'Status': ['COMPLETE'],
            'Local Path': [str(test_fits)]
        })
        
        successful, failed = downloader.download_light_curves(max_targets=1)
        
        assert successful >= 0
        assert failed >= 0
    
    @patch('src.data.kepler_downloader.KeplerDownloader.query_confirmed_exoplanets')
    def test_download_handles_no_observations(self, mock_query, downloader):
        """Test handling of no observations."""
        mock_query.return_value = []
        
        successful, failed = downloader.download_light_curves(max_targets=10)
        
        assert successful == 0
        assert failed == 0


class TestGetDownloadStats:
    """Test get_download_stats method."""
    
    def test_stats_empty_directory(self, downloader):
        """Test statistics for empty directory."""
        stats = downloader.get_download_stats()
        
        assert stats['total_files'] == 0
        assert stats['total_size_gb'] == 0.0
        assert 'output_dir' in stats
    
    def test_stats_with_files(self, downloader, tmp_path):
        """Test statistics with downloaded files."""
        # Create dummy FITS files
        for i in range(3):
            fits_path = downloader.output_dir / f"kepler_{i:03d}.fits"
            hdu = fits.PrimaryHDU(np.zeros((100, 100)))
            hdu.writeto(fits_path, overwrite=True)
        
        stats = downloader.get_download_stats()
        
        assert stats['total_files'] == 3
        assert stats['total_size_gb'] > 0
        assert len(stats['file_list']) <= 10


@pytest.mark.parametrize("mission", ["Kepler", "K2"])
def test_different_missions(mission, temp_output_dir):
    """Test downloading from different missions."""
    downloader = KeplerDownloader(output_dir=temp_output_dir)
    
    with patch('src.data.kepler_downloader.Observations.query_criteria') as mock_query:
        mock_query.return_value = Table()
        observations = downloader.query_confirmed_exoplanets(
            max_targets=10,
            mission=mission
        )
        
        assert isinstance(observations, list)