"""
TESS Exoplanet Data Downloader

This module downloads TESS (Transiting Exoplanet Survey Satellite) exoplanet data
via the MAST API. It fetches light curve data for TOI (TESS Objects of Interest),
parses JSON responses, extracts stellar parameters, and saves data in structured format.

Example:
    >>> downloader = TESSDownloader(output_dir="data/raw/tess")
    >>> downloader.download_toi_data(max_targets=150)
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import numpy as np
import pandas as pd
from astropy.io import fits
from astroquery.mast import Catalogs, Observations
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TESSDownloader:
    """
    Download and process TESS exoplanet data from MAST archive.
    
    This downloader queries the TESS Input Catalog (TIC) for exoplanet candidates
    (TOIs), downloads light curve data, and extracts stellar parameters. It includes
    API rate limiting and retry logic for robust operations.
    
    Attributes:
        output_dir (Path): Directory to save downloaded data
        api_base_url (str): Base URL for MAST API
        rate_limit_delay (float): Delay between API requests (seconds)
        max_retries (int): Maximum retry attempts for failed requests
    """
    
    def __init__(
        self,
        output_dir: str = "data/raw/tess",
        rate_limit_delay: float = 0.5,
        max_retries: int = 3
    ):
        """
        Initialize the TESS downloader.
        
        Args:
            output_dir: Directory to save downloaded light curves and metadata
            rate_limit_delay: Delay between API requests to avoid rate limiting
            max_retries: Maximum number of retry attempts per request
        """
        self.output_dir = Path(output_dir)
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries
        
        # Create output subdirectories
        self.fits_dir = self.output_dir / "fits"
        self.metadata_dir = self.output_dir / "metadata"
        self.fits_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup API session with retry logic
        self.session = self._create_retry_session()
        
        logger.info(f"Initialized TESSDownloader")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Rate limit delay: {rate_limit_delay}s")
    
    def _create_retry_session(self) -> requests.Session:
        """
        Create requests session with retry strategy.
        
        Returns:
            Configured requests Session object
        """
        session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def query_toi_catalog(self, max_targets: int = 150) -> pd.DataFrame:
        """
        Query TESS Objects of Interest (TOI) catalog.
        
        Args:
            max_targets: Maximum number of TOIs to retrieve
        
        Returns:
            DataFrame containing TOI metadata
        """
        logger.info(f"Querying TOI catalog for up to {max_targets} targets...")
        
        try:
            # Query TIC using astroquery
            catalog_data = Catalogs.query_criteria(
                catalog="Tic",
                objType="STAR"
            )
            
            if len(catalog_data) == 0:
                logger.warning("No TOI data found")
                return pd.DataFrame()
            
            # Convert to pandas DataFrame
            df = catalog_data.to_pandas()
            
            # Limit to max_targets
            df = df.head(max_targets)
            
            logger.info(f"Retrieved {len(df)} TOI candidates")
            return df
            
        except Exception as e:
            logger.error(f"Error querying TOI catalog: {e}")
            return pd.DataFrame()
    
    def download_toi_data(
        self,
        max_targets: int = 150,
        download_fits: bool = True
    ) -> Tuple[int, int]:
        """
        Download TESS light curve data for TOI candidates.
        
        Args:
            max_targets: Maximum number of TOIs to download
            download_fits: Whether to download FITS light curve files
        
        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        logger.info(f"Starting download of {max_targets} TESS TOI light curves")
        
        # Query TOI catalog
        toi_df = self.query_toi_catalog(max_targets=max_targets)
        
        if toi_df.empty:
            logger.error("No TOI data to download")
            return 0, 0
        
        successful = 0
        failed = 0
        
        # Download data for each TOI
        for idx, toi in tqdm(toi_df.iterrows(), total=len(toi_df), desc="Downloading TOI data"):
            try:
                # Extract TIC ID
                tic_id = toi.get('ID', f'TIC_{idx}')
                
                # Query observations for this TIC
                observations = self._query_tess_observations(tic_id)
                
                if not observations:
                    logger.debug(f"No observations for TIC {tic_id}")
                    failed += 1
                    continue
                
                # Extract and save stellar parameters
                stellar_params = self._extract_stellar_parameters(toi)
                self._save_metadata(tic_id, stellar_params, observations)
                
                # Download FITS light curve if requested
                if download_fits:
                    fits_success = self._download_fits_lightcurve(tic_id, observations)
                    if fits_success:
                        successful += 1
                    else:
                        failed += 1
                else:
                    successful += 1
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error processing TIC {tic_id}: {e}")
                failed += 1
        
        logger.info(f"Download complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def _query_tess_observations(self, tic_id: str) -> List[Dict[str, Any]]:
        """
        Query TESS observations for a specific TIC ID.
        
        Args:
            tic_id: TESS Input Catalog ID
        
        Returns:
            List of observation metadata dictionaries
        """
        try:
            obs_table = Observations.query_criteria(
                target_name=str(tic_id),
                obs_collection="TESS",
                dataproduct_type="timeseries"
            )
            
            if len(obs_table) == 0:
                return []
            
            # Convert to list of dicts
            observations = []
            for row in obs_table:
                obs_dict = {
                    'obs_id': row['obsid'],
                    'target_name': row['target_name'],
                    'sector': row.get('sequence_number', 0),
                    's_ra': row.get('s_ra', 0.0),
                    's_dec': row.get('s_dec', 0.0),
                    't_exptime': row.get('t_exptime', 0.0)
                }
                observations.append(obs_dict)
            
            return observations
            
        except Exception as e:
            logger.error(f"Error querying observations for TIC {tic_id}: {e}")
            return []
    
    def _extract_stellar_parameters(self, toi_row: pd.Series) -> Dict[str, Any]:
        """
        Extract stellar parameters from TOI catalog row.
        
        Args:
            toi_row: DataFrame row containing TOI data
        
        Returns:
            Dictionary of stellar parameters
        """
        stellar_params = {
            'tic_id': toi_row.get('ID', 'unknown'),
            'ra': toi_row.get('ra', 0.0),
            'dec': toi_row.get('dec', 0.0),
            'teff': toi_row.get('Teff', 5778.0),  # Effective temperature (K)
            'logg': toi_row.get('logg', 4.44),     # Surface gravity
            'radius': toi_row.get('rad', 1.0),     # Stellar radius (solar radii)
            'mass': toi_row.get('mass', 1.0),      # Stellar mass (solar masses)
            'tmag': toi_row.get('Tmag', 99.0),     # TESS magnitude
            'distance': toi_row.get('d', 0.0)      # Distance (parsecs)
        }
        
        return stellar_params
    
    def _save_metadata(
        self,
        tic_id: str,
        stellar_params: Dict[str, Any],
        observations: List[Dict[str, Any]]
    ):
        """
        Save TOI metadata to JSON file.
        
        Args:
            tic_id: TESS Input Catalog ID
            stellar_params: Stellar parameter dictionary
            observations: List of observation metadata
        """
        metadata = {
            'tic_id': tic_id,
            'stellar_parameters': stellar_params,
            'observations': observations,
            'download_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        
        output_file = self.metadata_dir / f"{tic_id}_metadata.json"
        with open(output_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.debug(f"Saved metadata: {output_file.name}")
    
    def _download_fits_lightcurve(
        self,
        tic_id: str,
        observations: List[Dict[str, Any]]
    ) -> bool:
        """
        Download FITS light curve file for a TIC ID.
        
        Args:
            tic_id: TESS Input Catalog ID
            observations: List of observation metadata
        
        Returns:
            True if download successful, False otherwise
        """
        try:
            if not observations:
                return False
            
            # Get first observation's data products
            obs_id = observations[0]['obs_id']
            products = Observations.get_product_list(obs_id)
            
            # Filter for light curve products
            lc_products = products[products['productSubGroupDescription'] == 'LC']
            
            if len(lc_products) == 0:
                logger.debug(f"No light curve products for TIC {tic_id}")
                return False
            
            # Download first light curve
            manifest = Observations.download_products(
                lc_products[:1],
                download_dir=str(self.fits_dir)
            )
            
            # Check if download was successful
            for file_row in manifest:
                if file_row['Status'] == 'COMPLETE':
                    logger.debug(f"Downloaded FITS: {tic_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error downloading FITS for TIC {tic_id}: {e}")
            return False
    
    def get_download_stats(self) -> Dict[str, Any]:
        """
        Get statistics about downloaded files.
        
        Returns:
            Dictionary with download statistics
        """
        fits_files = list(self.fits_dir.glob("**/*.fits"))
        metadata_files = list(self.metadata_dir.glob("*.json"))
        
        total_size = sum(f.stat().st_size for f in fits_files)
        
        stats = {
            'total_fits_files': len(fits_files),
            'total_metadata_files': len(metadata_files),
            'total_size_gb': total_size / (1024**3),
            'fits_dir': str(self.fits_dir),
            'metadata_dir': str(self.metadata_dir)
        }
        
        return stats


def main():
    """
    Command-line interface for TESS downloader.
    
    Usage:
        python tess_downloader.py --max-targets 150 --output data/raw/tess
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download TESS exoplanet data from MAST"
    )
    parser.add_argument(
        '--max-targets',
        type=int,
        default=150,
        help='Maximum number of TOIs to download (default: 150)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/tess',
        help='Output directory for TESS data'
    )
    parser.add_argument(
        '--no-fits',
        action='store_true',
        help='Skip downloading FITS files (metadata only)'
    )
    parser.add_argument(
        '--rate-limit',
        type=float,
        default=0.5,
        help='Delay between API requests (seconds)'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = TESSDownloader(
        output_dir=args.output,
        rate_limit_delay=args.rate_limit
    )
    
    # Download TOI data
    successful, failed = downloader.download_toi_data(
        max_targets=args.max_targets,
        download_fits=not args.no_fits
    )
    
    # Print statistics
    stats = downloader.get_download_stats()
    print("\n=== Download Statistics ===")
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total FITS files: {stats['total_fits_files']}")
    print(f"Total metadata files: {stats['total_metadata_files']}")
    print(f"Total size: {stats['total_size_gb']:.2f} GB")
    print(f"FITS directory: {stats['fits_dir']}")
    print(f"Metadata directory: {stats['metadata_dir']}")


if __name__ == "__main__":
    main()