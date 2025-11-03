"""
Unit tests for ZTF downloader module.

These tests mock external API calls and file operations.
"""

import pytest
import hashlib
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.data.ztf_downloader import ZTFDownloader


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "ztf_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir)


@pytest.fixture
def downloader(temp_output_dir):
    """Create a ZTFDownloader instance for testing."""
    return ZTFDownloader(
        output_dir=temp_output_dir,
        verify_checksum=True,
        resume_downloads=True,
        max_retries=2
    )


class TestZTFDownloaderInit:
    """Test ZTFDownloader initialization."""
    
    def test_init_creates_directories(self, temp_output_dir):
        """Test that initialization creates required directories."""
        downloader = ZTFDownloader(output_dir=temp_output_dir)
        
        assert downloader.output_dir.exists()
        assert downloader.images_dir.exists()
        assert downloader.metadata_dir.exists()
        assert downloader.temp_dir.exists()
    
    def test_init_with_custom_chunk_size(self, temp_output_dir):
        """Test initialization with custom chunk size."""
        chunk_size = 1024 * 1024  # 1MB
        downloader = ZTFDownloader(
            output_dir=temp_output_dir,
            chunk_size=chunk_size
        )
        
        assert downloader.chunk_size == chunk_size


class TestQueryAvailableImages:
    """Test query_available_images method."""
    
    def test_query_generates_image_list(self, downloader):
        """Test that query generates list of images."""
        images = downloader.query_available_images(
            start_date="2024-01-01",
            end_date="2024-01-03",
            max_images=50
        )
        
        assert isinstance(images, list)
        assert len(images) > 0
        assert all('filename' in img for img in images)
        assert all('field_id' in img for img in images)
    
    def test_query_respects_max_images(self, downloader):
        """Test that query respects max_images limit."""
        max_images = 10
        images = downloader.query_available_images(
            start_date="2024-01-01",
            end_date="2024-01-31",
            max_images=max_images
        )
        
        assert len(images) <= max_images
    
    def test_query_filters_by_field_ids(self, downloader):
        """Test filtering by specific field IDs."""
        field_ids = [100, 200]
        images = downloader.query_available_images(
            start_date="2024-01-01",
            end_date="2024-01-02",
            field_ids=field_ids,
            max_images=20
        )
        
        # Check that only specified fields are included
        for img in images:
            assert img['field_id'] in field_ids
    
    def test_query_handles_invalid_dates(self, downloader):
        """Test handling of invalid date formats."""
        with pytest.raises(ValueError):
            downloader.query_available_images(
                start_date="invalid-date",
                end_date="2024-01-31",
                max_images=10
            )


class TestDownloadImages:
    """Test download_images method."""
    
    @patch('src.data.ztf_downloader.ZTFDownloader._download_single_image')
    @patch('src.data.ztf_downloader.ZTFDownloader.query_available_images')
    def test_download_success(self, mock_query, mock_download, downloader):
        """Test successful download of images."""
        # Mock image query
        mock_images = [
            {'filename': 'ztf_001.fits', 'url': 'http://example.com/ztf_001.fits'},
            {'filename': 'ztf_002.fits', 'url': 'http://example.com/ztf_002.fits'}
        ]
        mock_query.return_value = mock_images
        
        # Mock successful downloads
        mock_download.return_value = True
        
        successful, failed = downloader.download_images(
            start_date="2024-01-01",
            end_date="2024-01-02",
            max_images=2
        )
        
        assert successful == 2
        assert failed == 0
    
    @patch('src.data.ztf_downloader.ZTFDownloader.query_available_images')
    def test_download_handles_no_images(self, mock_query, downloader):
        """Test handling of no available images."""
        mock_query.return_value = []
        
        successful, failed = downloader.download_images(
            start_date="2024-01-01",
            end_date="2024-01-02",
            max_images=10
        )
        
        assert successful == 0
        assert failed == 0


class TestDownloadSingleImage:
    """Test _download_single_image method."""
    
    def test_skip_existing_file(self, downloader):
        """Test that existing files are skipped."""
        # Create an existing file
        existing_file = downloader.images_dir / "ztf_existing.fits"
        existing_file.touch()
        
        image_meta = {
            'filename': 'ztf_existing.fits',
            'url': 'http://example.com/ztf_existing.fits',
            'size_mb': 150
        }
        
        result = downloader._download_single_image(image_meta)
        
        assert result is True
    
    def test_creates_placeholder_fits(self, downloader):
        """Test placeholder FITS creation."""
        image_meta = {
            'filename': 'ztf_new.fits',
            'url': 'http://example.com/ztf_new.fits',
            'size_mb': 150
        }
        
        result = downloader._download_single_image(image_meta)
        
        output_file = downloader.images_dir / 'ztf_new.fits'
        assert output_file.exists()


class TestVerifyFileChecksum:
    """Test _verify_file_checksum method."""
    
    def test_verify_correct_checksum(self, downloader, tmp_path):
        """Test verification with correct checksum."""
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_content = b"test content for checksum"
        test_file.write_bytes(test_content)
        
        # Calculate expected MD5
        expected_md5 = hashlib.md5(test_content).hexdigest()
        
        result = downloader._verify_file_checksum(test_file, expected_md5)
        
        assert result is True
    
    def test_verify_incorrect_checksum(self, downloader, tmp_path):
        """Test verification with incorrect checksum."""
        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"test content")
        
        wrong_md5 = "0" * 32  # Invalid MD5
        
        result = downloader._verify_file_checksum(test_file, wrong_md5)
        
        assert result is False


class TestCheckDiskSpace:
    """Test check_disk_space method."""
    
    def test_sufficient_disk_space(self, downloader):
        """Test with sufficient disk space."""
        # Request a small amount
        result = downloader.check_disk_space(required_gb=0.001)
        
        assert result is True
    
    @patch('shutil.disk_usage')
    def test_insufficient_disk_space(self, mock_usage, downloader):
        """Test with insufficient disk space."""
        # Mock very low disk space
        mock_usage.return_value = Mock(free=1024 * 1024)  # 1MB
        
        result = downloader.check_disk_space(required_gb=100.0)
        
        assert result is False


class TestGetDownloadStats:
    """Test get_download_stats method."""
    
    def test_stats_empty_directory(self, downloader):
        """Test statistics for empty directory."""
        stats = downloader.get_download_stats()
        
        assert stats['total_images'] == 0
        assert stats['total_size_gb'] == 0.0
        assert 'images_dir' in stats
    
    def test_stats_with_images(self, downloader):
        """Test statistics with downloaded images."""
        # Create dummy FITS files
        from astropy.io import fits
        import numpy as np
        
        for i in range(3):
            fits_path = downloader.images_dir / f"ztf_{i:03d}.fits"
            hdu = fits.PrimaryHDU(np.zeros((100, 100)))
            hdu.writeto(fits_path, overwrite=True)
        
        stats = downloader.get_download_stats()
        
        assert stats['total_images'] == 3
        assert stats['total_size_gb'] > 0


class TestCleanupTemp:
    """Test _cleanup_temp method."""
    
    def test_cleanup_removes_partial_files(self, downloader):
        """Test that cleanup removes .part files."""
        # Create dummy .part files
        for i in range(3):
            part_file = downloader.temp_dir / f"ztf_{i}.fits.part"
            part_file.touch()
        
        downloader._cleanup_temp()
        
        # Check that .part files are removed
        part_files = list(downloader.temp_dir.glob("*.part"))
        assert len(part_files) == 0


@pytest.mark.parametrize("chunk_size", [1024, 8192, 65536])
def test_different_chunk_sizes(chunk_size, temp_output_dir):
    """Test initialization with different chunk sizes."""
    downloader = ZTFDownloader(
        output_dir=temp_output_dir,
        chunk_size=chunk_size
    )
    
    assert downloader.chunk_size == chunk_size


def test_resumable_downloads_disabled(temp_output_dir):
    """Test with resumable downloads disabled."""
    downloader = ZTFDownloader(
        output_dir=temp_output_dir,
        resume_downloads=False
    )
    
    assert downloader.resume_downloads is False