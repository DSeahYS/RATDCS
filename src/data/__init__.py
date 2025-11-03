"""
RATDCS Data Acquisition Module

This module provides data downloaders for astronomical datasets used in RATDCS.
It includes downloaders for Kepler exoplanet light curves, TESS exoplanet data,
and ZTF asteroid survey images.

Available Downloaders:
    - KeplerDownloader: Download Kepler/K2 exoplanet light curves from MAST
    - TESSDownloader: Download TESS exoplanet data and TOI catalog
    - ZTFDownloader: Download ZTF asteroid survey FITS images

Example:
    >>> from src.data import KeplerDownloader, TESSDownloader, ZTFDownloader
    >>> 
    >>> # Download Kepler data
    >>> kepler = KeplerDownloader(output_dir="data/raw/kepler")
    >>> kepler.download_light_curves(max_targets=100)
    >>> 
    >>> # Download TESS data
    >>> tess = TESSDownloader(output_dir="data/raw/tess")
    >>> tess.download_toi_data(max_targets=150)
    >>> 
    >>> # Download ZTF data
    >>> ztf = ZTFDownloader(output_dir="data/raw/ztf")
    >>> ztf.download_images(start_date="2024-01-01", end_date="2024-01-31")
"""

from .kepler_downloader import KeplerDownloader
from .tess_downloader import TESSDownloader
from .ztf_downloader import ZTFDownloader

__all__ = [
    'KeplerDownloader',
    'TESSDownloader',
    'ZTFDownloader',
]

__version__ = '1.0.0'
__author__ = 'Dave Seah Yong Sheng, NTU Aerospace Engineering'