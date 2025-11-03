"""
Kepler Exoplanet Light Curve Downloader

This module downloads Kepler exoplanet light curves from the MAST archive using astroquery.
It queries confirmed exoplanets, downloads FITS time-series data, verifies integrity,
and removes corrupted files.

Example:
    >>> downloader = KeplerDownloader(output_dir="data/raw/kepler")
    >>> downloader.download_light_curves(max_targets=100)
"""

import logging
import os
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import yaml

import numpy as np
from astropy.io import fits
from astroquery.mast import Observations
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KeplerDownloader:
    """
    Download and validate Kepler exoplanet light curves from MAST archive.
    
    This downloader queries the MAST archive for confirmed Kepler exoplanet
    observations, downloads FITS files containing light curve data, and performs
    integrity validation to ensure data quality.
    
    Attributes:
        output_dir (Path): Directory to save downloaded FITS files
        cache_dir (Path): Directory for temporary downloads
        verify_fits (bool): Whether to verify FITS file integrity
        max_retries (int): Maximum download retry attempts
    """
    
    def __init__(
        self,
        output_dir: str = "data/raw/kepler",
        cache_dir: Optional[str] = None,
        verify_fits: bool = True,
        max_retries: int = 3
    ):
        """
        Initialize the Kepler downloader.
        
        Args:
            output_dir: Directory to save downloaded light curves
            cache_dir: Temporary directory for downloads (default: output_dir/cache)
            verify_fits: Enable FITS integrity verification
            max_retries: Maximum number of download attempts per file
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.output_dir / "cache"
        self.verify_fits = verify_fits
        self.max_retries = max_retries
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized KeplerDownloader")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def query_confirmed_exoplanets(
        self,
        max_targets: int = 100,
        mission: str = "Kepler"
    ) -> List[Dict]:
        """
        Query MAST for confirmed exoplanet observations.
        
        Args:
            max_targets: Maximum number of targets to query
            mission: Mission name ("Kepler" or "K2")
        
        Returns:
            List of observation metadata dictionaries
        """
        logger.info(f"Querying MAST for {mission} confirmed exoplanet observations...")
        
        try:
            # Query observations with LIGHTCURVE product type
            obs_table = Observations.query_criteria(
                obs_collection=mission,
                dataproduct_type=["timeseries"],
                intentType="science",
                dataRights="PUBLIC"
            )
            
            if len(obs_table) == 0:
                logger.warning(f"No observations found for {mission}")
                return []
            
            # Limit to max_targets
            obs_table = obs_table[:max_targets]
            
            logger.info(f"Found {len(obs_table)} observations")
            
            # Convert to list of dicts
            observations = []
            for row in obs_table:
                obs_dict = {
                    'obs_id': row['obsid'],
                    'target_name': row['target_name'],
                    'obs_collection': row['obs_collection'],
                    's_ra': row.get('s_ra', 0.0),
                    's_dec': row.get('s_dec', 0.0)
                }
                observations.append(obs_dict)
            
            return observations
            
        except Exception as e:
            logger.error(f"Error querying MAST: {e}")
            return []
    
    def download_light_curves(
        self,
        max_targets: int = 100,
        mission: str = "Kepler"
    ) -> Tuple[int, int]:
        """
        Download light curve FITS files for confirmed exoplanets.
        
        Args:
            max_targets: Maximum number of targets to download (MVP: 100+)
            mission: Mission name ("Kepler" or "K2")
        
        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        logger.info(f"Starting download of {max_targets} {mission} light curves")
        
        # Query observations
        observations = self.query_confirmed_exoplanets(
            max_targets=max_targets,
            mission=mission
        )
        
        if not observations:
            logger.error("No observations to download")
            return 0, 0
        
        successful = 0
        failed = 0
        
        # Download each observation's data products
        for obs in tqdm(observations, desc="Downloading light curves"):
            try:
                # Get data products for this observation
                products = Observations.get_product_list(obs['obs_id'])
                
                # Filter for light curve FITS files
                lc_products = products[
                    (products['productSubGroupDescription'] == 'LLC') |
                    (products['productSubGroupDescription'] == 'LC')
                ]
                
                if len(lc_products) == 0:
                    logger.debug(f"No light curve products for {obs['target_name']}")
                    failed += 1
                    continue
                
                # Download the first light curve product
                manifest = Observations.download_products(
                    lc_products[:1],
                    download_dir=str(self.cache_dir)
                )
                
                # Process downloaded files
                for file_row in manifest:
                    if file_row['Status'] == 'COMPLETE':
                        src_path = Path(file_row['Local Path'])
                        
                        # Verify FITS integrity
                        if self.verify_fits:
                            if not self._verify_fits_file(src_path):
                                logger.warning(f"Corrupted FITS file: {src_path.name}")
                                src_path.unlink()
                                failed += 1
                                continue
                        
                        # Move to output directory
                        dest_path = self.output_dir / src_path.name
                        shutil.move(str(src_path), str(dest_path))
                        
                        logger.debug(f"Downloaded: {dest_path.name}")
                        successful += 1
                    else:
                        logger.warning(f"Download failed for {obs['target_name']}")
                        failed += 1
                        
            except Exception as e:
                logger.error(f"Error downloading {obs['target_name']}: {e}")
                failed += 1
        
        # Clean up cache directory
        self._cleanup_cache()
        
        logger.info(f"Download complete: {successful} successful, {failed} failed")
        return successful, failed
    
    def _verify_fits_file(self, fits_path: Path) -> bool:
        """
        Verify FITS file integrity.
        
        Args:
            fits_path: Path to FITS file
        
        Returns:
            True if file is valid, False otherwise
        """
        try:
            with fits.open(fits_path, memmap=False) as hdul:
                # Check if file has data
                if len(hdul) < 2:
                    return False
                
                # Try to access primary data
                data = hdul[1].data
                if data is None or len(data) == 0:
                    return False
                
                # Check for required columns (Kepler light curves)
                required_cols = ['TIME', 'PDCSAP_FLUX']
                if not all(col in data.columns.names for col in required_cols):
                    return False
                
                # Check for NaN values
                time = data['TIME']
                flux = data['PDCSAP_FLUX']
                if np.all(np.isnan(time)) or np.all(np.isnan(flux)):
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"FITS verification error for {fits_path.name}: {e}")
            return False
    
    def _cleanup_cache(self):
        """Remove temporary cache directory."""
        try:
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                logger.debug("Cache directory cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning cache: {e}")
    
    def get_download_stats(self) -> Dict[str, any]:
        """
        Get statistics about downloaded files.
        
        Returns:
            Dictionary with download statistics
        """
        fits_files = list(self.output_dir.glob("*.fits"))
        total_size = sum(f.stat().st_size for f in fits_files)
        
        stats = {
            'total_files': len(fits_files),
            'total_size_gb': total_size / (1024**3),
            'output_dir': str(self.output_dir),
            'file_list': [f.name for f in fits_files[:10]]  # First 10
        }
        
        return stats


def main():
    """
    Command-line interface for Kepler downloader.
    
    Usage:
        python kepler_downloader.py --max-targets 100 --output data/raw/kepler
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Kepler exoplanet light curves from MAST"
    )
    parser.add_argument(
        '--max-targets',
        type=int,
        default=100,
        help='Maximum number of targets to download (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/kepler',
        help='Output directory for FITS files'
    )
    parser.add_argument(
        '--mission',
        type=str,
        default='Kepler',
        choices=['Kepler', 'K2'],
        help='Mission to download from'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Disable FITS integrity verification'
    )
    
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = KeplerDownloader(
        output_dir=args.output,
        verify_fits=not args.no_verify
    )
    
    # Download light curves
    successful, failed = downloader.download_light_curves(
        max_targets=args.max_targets,
        mission=args.mission
    )
    
    # Print statistics
    stats = downloader.get_download_stats()
    print("\n=== Download Statistics ===")
    print(f"Successful downloads: {successful}")
    print(f"Failed downloads: {failed}")
    print(f"Total files: {stats['total_files']}")
    print(f"Total size: {stats['total_size_gb']:.2f} GB")
    print(f"Output directory: {stats['output_dir']}")


if __name__ == "__main__":
    main()