"""
Unit tests for TESS downloader module.

These tests mock external API calls to avoid actual data downloads.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from astropy.table import Table

from src.data.tess_downloader import TESSDownloader


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "tess_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


@pytest.fixture
def downloader(temp_output_dir):
    """Create a TESSDownloader instance for testing."""
    return TESSDownloader(
        output_dir=temp_output_dir,
        rate_limit_delay=0.1,
        max_retries=2
    )


class TestTESSDownloaderInit:
    """Test TESSDownloader initialization."""
    
    def test_init_creates_directories(self, temp_output_dir):
        """Test that initialization creates required directories."""
        downloader = TESSDownloader(output_dir=temp_output_dir)
        
        assert downloader.output_dir.exists()
        assert downloader.fits_dir.exists()
        assert downloader.metadata_dir.exists()
    
    def test_init_creates_retry_session(self, downloader):
        """Test that retry session is created."""
        assert downloader.session is not None
        assert hasattr(downloader.session, 'mount')


class TestQueryTOICatalog:
    """Test query_toi_catalog method."""
    
    @patch('src.data.tess_downloader.Catalogs.query_criteria')
    def test_query_returns_dataframe(self, mock_query, downloader):
        """Test that query returns pandas DataFrame."""
        # Mock catalog data
        mock_table = Table({
            'ID': ['TIC1', 'TIC2', 'TIC3'],
            'ra': [290.0, 291.0, 292.0],
            'dec': [45.0, 46.0, 47.0],
            'Teff': [5778.0, 6000.0, 5500.0],
            'rad': [1.0, 1.2, 0.9]
        })
        mock_query.return_value = mock_table
        
        df = downloader.query_toi_catalog(max_targets=3)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 3
    
    @patch('src.data.tess_downloader.Catalogs.query_criteria')
    def test_query_handles_empty_results(self, mock_query, downloader):
        """Test handling of empty query results."""
        mock_query.return_value = Table()
        
        df = downloader.query_toi_catalog(max_targets=10)
        
        assert df.empty


class TestExtractStellarParameters:
    """Test _extract_stellar_parameters method."""
    
    def test_extract_basic_parameters(self, downloader):
        """Test extraction of stellar parameters from TOI row."""
        toi_row = pd.Series({
            'ID': 'TIC12345',
            'ra': 290.5,
            'dec': 45.2,
            'Teff': 5778.0,
            'logg': 4.44,
            'rad': 1.0,
            'mass': 1.0,
            'Tmag': 10.5,
            'd': 100.0
        })
        
        params = downloader._extract_stellar_parameters(toi_row)
        
        assert params['tic_id'] == 'TIC12345'
        assert params['teff'] == 5778.0
        assert params['logg'] == 4.44
        assert params['distance'] == 100.0
    
    def test_extract_uses_defaults_for_missing(self, downloader):
        """Test that missing values use reasonable defaults."""
        toi_row = pd.Series({'ID': 'TIC99999'})
        
        params = downloader._extract_stellar_parameters(toi_row)
        
        assert params['tic_id'] == 'TIC99999'
        assert params['teff'] == 5778.0  # Solar default
        assert params['logg'] == 4.44  # Solar default


class TestSaveMetadata:
    """Test _save_metadata method."""
    
    def test_metadata_saved_as_json(self, downloader):
        """Test that metadata is saved as JSON file."""
        tic_id = 'TIC12345'
        stellar_params = {
            'tic_id': tic_id,
            'teff': 5778.0,
            'ra': 290.0,
            'dec': 45.0
        }
        observations = [
            {'obs_id': 'obs1', 'sector': 1}
        ]
        
        downloader._save_metadata(tic_id, stellar_params, observations)
        
        metadata_file = downloader.metadata_dir / f"{tic_id}_metadata.json"
        assert metadata_file.exists()
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        assert data['tic_id'] == tic_id
        assert 'stellar_parameters' in data
        assert 'observations' in data
        assert 'download_timestamp' in data


class TestQueryTESSObservations:
    """Test _query_tess_observations method."""
    
    @patch('src.data.tess_downloader.Observations.query_criteria')
    def test_query_observations_returns_list(self, mock_query, downloader):
        """Test that observations are returned as list."""
        mock_table = Table({
            'obsid': ['obs1', 'obs2'],
            'target_name': ['TIC12345', 'TIC12345'],
            'sequence_number': [1, 2],
            's_ra': [290.0, 290.0],
            's_dec': [45.0, 45.0]
        })
        mock_query.return_value = mock_table
        
        observations = downloader._query_tess_observations('TIC12345')
        
        assert isinstance(observations, list)
        assert len(observations) == 2
        assert observations[0]['sector'] == 1
    
    @patch('src.data.tess_downloader.Observations.query_criteria')
    def test_query_handles_no_observations(self, mock_query, downloader):
        """Test handling of missing observations."""
        mock_query.return_value = Table()
        
        observations = downloader._query_tess_observations('TIC99999')
        
        assert len(observations) == 0


class TestDownloadTOIData:
    """Test download_toi_data method."""
    
    @patch('src.data.tess_downloader.TESSDownloader._download_fits_lightcurve')
    @patch('src.data.tess_downloader.TESSDownloader._query_tess_observations')
    @patch('src.data.tess_downloader.TESSDownloader.query_toi_catalog')
    def test_download_success(self, mock_catalog, mock_obs, mock_fits, downloader):
        """Test successful download of TOI data."""
        # Mock catalog query
        mock_df = pd.DataFrame({
            'ID': ['TIC1', 'TIC2'],
            'ra': [290.0, 291.0],
            'dec': [45.0, 46.0]
        })
        mock_catalog.return_value = mock_df
        
        # Mock observations
        mock_obs.return_value = [{'obs_id': 'obs1', 'sector': 1}]
        
        # Mock FITS download
        mock_fits.return_value = True
        
        successful, failed = downloader.download_toi_data(max_targets=2)
        
        assert successful + failed == 2
    
    @patch('src.data.tess_downloader.TESSDownloader.query_toi_catalog')
    def test_download_handles_empty_catalog(self, mock_catalog, downloader):
        """Test handling of empty catalog."""
        mock_catalog.return_value = pd.DataFrame()
        
        successful, failed = downloader.download_toi_data(max_targets=10)
        
        assert successful == 0
        assert failed == 0


class TestGetDownloadStats:
    """Test get_download_stats method."""
    
    def test_stats_empty_directory(self, downloader):
        """Test statistics for empty directories."""
        stats = downloader.get_download_stats()
        
        assert stats['total_fits_files'] == 0
        assert stats['total_metadata_files'] == 0
        assert stats['total_size_gb'] == 0.0
    
    def test_stats_with_metadata(self, downloader):
        """Test statistics with metadata files."""
        # Create dummy metadata files
        for i in range(3):
            metadata_file = downloader.metadata_dir / f"TIC{i}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({'tic_id': f'TIC{i}'}, f)
        
        stats = downloader.get_download_stats()
        
        assert stats['total_metadata_files'] == 3


@pytest.mark.parametrize("rate_limit", [0.1, 0.5, 1.0])
def test_different_rate_limits(rate_limit, temp_output_dir):
    """Test initialization with different rate limits."""
    downloader = TESSDownloader(
        output_dir=temp_output_dir,
        rate_limit_delay=rate_limit
    )
    
    assert downloader.rate_limit_delay == rate_limit