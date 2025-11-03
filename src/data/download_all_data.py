"""
RATDCS Master Data Download Script

This script orchestrates all three data downloaders (Kepler, TESS, ZTF) to download
astronomical data for RATDCS. It provides a CLI interface with options to download
specific datasets or all datasets, includes progress reporting, and checks disk space
before downloading.

Usage:
    # Download all datasets with defaults
    python download_all_data.py --all
    
    # Download specific datasets
    python download_all_data.py --kepler --tess --max-targets 100
    
    # Download ZTF data for date range
    python download_all_data.py --ztf --start-date 2024-01-01 --end-date 2024-01-31
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Any
import yaml

from .kepler_downloader import KeplerDownloader
from .tess_downloader import TESSDownloader
from .ztf_downloader import ZTFDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('data_download.log')
    ]
)
logger = logging.getLogger(__name__)


class DataDownloadOrchestrator:
    """
    Orchestrate downloads from multiple astronomical data sources.
    
    This class manages the download process for Kepler, TESS, and ZTF data,
    providing progress reporting, disk space checking, and error handling.
    
    Attributes:
        base_dir (Path): Base directory for all downloads
        config (Dict): Configuration dictionary
        kepler_downloader (KeplerDownloader): Kepler data downloader
        tess_downloader (TESSDownloader): TESS data downloader
        ztf_downloader (ZTFDownloader): ZTF data downloader
    """
    
    def __init__(
        self,
        base_dir: str = "data/raw",
        config_path: str = "config/default.yaml"
    ):
        """
        Initialize the data download orchestrator.
        
        Args:
            base_dir: Base directory for all data downloads
            config_path: Path to configuration file
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize downloaders
        self.kepler_downloader = KeplerDownloader(
            output_dir=str(self.base_dir / "kepler")
        )
        self.tess_downloader = TESSDownloader(
            output_dir=str(self.base_dir / "tess")
        )
        self.ztf_downloader = ZTFDownloader(
            output_dir=str(self.base_dir / "ztf")
        )
        
        logger.info(f"Initialized DataDownloadOrchestrator")
        logger.info(f"Base directory: {self.base_dir}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        
        Returns:
            Configuration dictionary
        """
        try:
            config_file = Path(config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded configuration from {config_path}")
                    return config
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
        
        # Return default configuration
        return {
            'data_pipeline': {
                'preprocessing': {
                    'normalize': True,
                    'cache_enabled': True
                }
            }
        }
    
    def check_disk_space(self, required_gb: float) -> bool:
        """
        Check if sufficient disk space is available.
        
        Args:
            required_gb: Required space in gigabytes
        
        Returns:
            True if sufficient space available, False otherwise
        """
        try:
            stat = shutil.disk_usage(self.base_dir)
            available_gb = stat.free / (1024**3)
            
            logger.info(f"=== Disk Space Check ===")
            logger.info(f"Available: {available_gb:.2f} GB")
            logger.info(f"Required: {required_gb:.2f} GB")
            
            if available_gb < required_gb:
                logger.error(f"Insufficient disk space! Need {required_gb:.2f} GB, "
                           f"but only {available_gb:.2f} GB available")
                return False
            
            logger.info(f"Disk space check passed")
            return True
            
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return False
    
    def download_kepler_data(
        self,
        max_targets: int = 100,
        mission: str = "Kepler"
    ) -> Dict[str, int]:
        """
        Download Kepler exoplanet light curves.
        
        Args:
            max_targets: Maximum number of targets to download
            mission: Mission name ("Kepler" or "K2")
        
        Returns:
            Dictionary with download statistics
        """
        logger.info("=" * 60)
        logger.info(f"Starting Kepler data download ({mission})")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            successful, failed = self.kepler_downloader.download_light_curves(
                max_targets=max_targets,
                mission=mission
            )
            
            elapsed_time = time.time() - start_time
            stats = self.kepler_downloader.get_download_stats()
            
            logger.info(f"Kepler download completed in {elapsed_time:.1f} seconds")
            
            return {
                'successful': successful,
                'failed': failed,
                'total_files': stats['total_files'],
                'total_size_gb': stats['total_size_gb'],
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error during Kepler download: {e}")
            return {'successful': 0, 'failed': max_targets, 'error': str(e)}
    
    def download_tess_data(
        self,
        max_targets: int = 150
    ) -> Dict[str, int]:
        """
        Download TESS exoplanet data.
        
        Args:
            max_targets: Maximum number of TOIs to download
        
        Returns:
            Dictionary with download statistics
        """
        logger.info("=" * 60)
        logger.info(f"Starting TESS data download")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            successful, failed = self.tess_downloader.download_toi_data(
                max_targets=max_targets
            )
            
            elapsed_time = time.time() - start_time
            stats = self.tess_downloader.get_download_stats()
            
            logger.info(f"TESS download completed in {elapsed_time:.1f} seconds")
            
            return {
                'successful': successful,
                'failed': failed,
                'total_fits_files': stats['total_fits_files'],
                'total_metadata_files': stats['total_metadata_files'],
                'total_size_gb': stats['total_size_gb'],
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error during TESS download: {e}")
            return {'successful': 0, 'failed': max_targets, 'error': str(e)}
    
    def download_ztf_data(
        self,
        start_date: str,
        end_date: str,
        field_ids: list = None,
        max_images: int = 100
    ) -> Dict[str, int]:
        """
        Download ZTF asteroid survey images.
        
        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            field_ids: List of ZTF field IDs to download
            max_images: Maximum number of images to download
        
        Returns:
            Dictionary with download statistics
        """
        logger.info("=" * 60)
        logger.info(f"Starting ZTF data download")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            successful, failed = self.ztf_downloader.download_images(
                start_date=start_date,
                end_date=end_date,
                field_ids=field_ids,
                max_images=max_images
            )
            
            elapsed_time = time.time() - start_time
            stats = self.ztf_downloader.get_download_stats()
            
            logger.info(f"ZTF download completed in {elapsed_time:.1f} seconds")
            
            return {
                'successful': successful,
                'failed': failed,
                'total_images': stats['total_images'],
                'total_size_gb': stats['total_size_gb'],
                'elapsed_time': elapsed_time
            }
            
        except Exception as e:
            logger.error(f"Error during ZTF download: {e}")
            return {'successful': 0, 'failed': max_images, 'error': str(e)}
    
    def download_all(
        self,
        kepler_targets: int = 100,
        tess_targets: int = 150,
        ztf_start_date: str = "2024-01-01",
        ztf_end_date: str = "2024-01-31",
        ztf_max_images: int = 100
    ) -> Dict[str, Any]:
        """
        Download all datasets.
        
        Args:
            kepler_targets: Number of Kepler targets to download
            tess_targets: Number of TESS targets to download
            ztf_start_date: ZTF start date
            ztf_end_date: ZTF end date
            ztf_max_images: Maximum ZTF images to download
        
        Returns:
            Dictionary with combined statistics
        """
        logger.info("=" * 60)
        logger.info("Starting complete data download (Kepler + TESS + ZTF)")
        logger.info("=" * 60)
        
        total_start_time = time.time()
        
        # Estimate required disk space
        # Kepler: ~30MB per target, TESS: ~150MB per target, ZTF: ~150MB per image
        estimated_gb = (kepler_targets * 30 + tess_targets * 150 + ztf_max_images * 150) / 1024
        
        if not self.check_disk_space(estimated_gb * 1.2):  # 20% buffer
            logger.error("Aborting download due to insufficient disk space")
            return {'success': False, 'reason': 'insufficient_disk_space'}
        
        results = {}
        
        # Download Kepler data
        logger.info("\n[1/3] Downloading Kepler data...")
        results['kepler'] = self.download_kepler_data(max_targets=kepler_targets)
        
        # Download TESS data
        logger.info("\n[2/3] Downloading TESS data...")
        results['tess'] = self.download_tess_data(max_targets=tess_targets)
        
        # Download ZTF data
        logger.info("\n[3/3] Downloading ZTF data...")
        results['ztf'] = self.download_ztf_data(
            start_date=ztf_start_date,
            end_date=ztf_end_date,
            max_images=ztf_max_images
        )
        
        total_elapsed = time.time() - total_start_time
        
        # Calculate totals
        total_successful = (
            results['kepler'].get('successful', 0) +
            results['tess'].get('successful', 0) +
            results['ztf'].get('successful', 0)
        )
        total_failed = (
            results['kepler'].get('failed', 0) +
            results['tess'].get('failed', 0) +
            results['ztf'].get('failed', 0)
        )
        total_size_gb = (
            results['kepler'].get('total_size_gb', 0) +
            results['tess'].get('total_size_gb', 0) +
            results['ztf'].get('total_size_gb', 0)
        )
        
        results['summary'] = {
            'total_successful': total_successful,
            'total_failed': total_failed,
            'total_size_gb': total_size_gb,
            'total_elapsed_time': total_elapsed
        }
        
        logger.info("=" * 60)
        logger.info("All downloads completed!")
        logger.info("=" * 60)
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]):
        """
        Print download summary.
        
        Args:
            results: Results dictionary from download operations
        """
        summary = results.get('summary', {})
        
        print("\n" + "=" * 60)
        print("DOWNLOAD SUMMARY")
        print("=" * 60)
        
        if 'kepler' in results:
            print(f"\nKepler:")
            print(f"  Successful: {results['kepler'].get('successful', 0)}")
            print(f"  Failed: {results['kepler'].get('failed', 0)}")
            print(f"  Size: {results['kepler'].get('total_size_gb', 0):.2f} GB")
        
        if 'tess' in results:
            print(f"\nTESS:")
            print(f"  Successful: {results['tess'].get('successful', 0)}")
            print(f"  Failed: {results['tess'].get('failed', 0)}")
            print(f"  Size: {results['tess'].get('total_size_gb', 0):.2f} GB")
        
        if 'ztf' in results:
            print(f"\nZTF:")
            print(f"  Successful: {results['ztf'].get('successful', 0)}")
            print(f"  Failed: {results['ztf'].get('failed', 0)}")
            print(f"  Size: {results['ztf'].get('total_size_gb', 0):.2f} GB")
        
        if summary:
            print(f"\nTotal Statistics:")
            print(f"  Total Successful: {summary.get('total_successful', 0)}")
            print(f"  Total Failed: {summary.get('total_failed', 0)}")
            print(f"  Total Size: {summary.get('total_size_gb', 0):.2f} GB")
            print(f"  Total Time: {summary.get('total_elapsed_time', 0):.1f} seconds")
        
        print("=" * 60)


def main():
    """
    Command-line interface for data download orchestrator.
    """
    parser = argparse.ArgumentParser(
        description="RATDCS Master Data Download Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets with defaults
  python download_all_data.py --all

  # Download specific datasets
  python download_all_data.py --kepler --tess --max-targets 100

  # Download ZTF data only
  python download_all_data.py --ztf --start-date 2024-01-01 --end-date 2024-01-31

  # Custom configuration
  python download_all_data.py --all --base-dir /mnt/data --kepler-targets 200
        """
    )
    
    # General options
    parser.add_argument(
        '--base-dir',
        type=str,
        default='data/raw',
        help='Base directory for all downloads (default: data/raw)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/default.yaml',
        help='Configuration file path'
    )
    
    # Dataset selection
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all datasets'
    )
    parser.add_argument(
        '--kepler',
        action='store_true',
        help='Download Kepler data'
    )
    parser.add_argument(
        '--tess',
        action='store_true',
        help='Download TESS data'
    )
    parser.add_argument(
        '--ztf',
        action='store_true',
        help='Download ZTF data'
    )
    
    # Kepler options
    parser.add_argument(
        '--kepler-targets',
        type=int,
        default=100,
        help='Number of Kepler targets to download (default: 100)'
    )
    parser.add_argument(
        '--kepler-mission',
        type=str,
        default='Kepler',
        choices=['Kepler', 'K2'],
        help='Kepler mission (default: Kepler)'
    )
    
    # TESS options
    parser.add_argument(
        '--tess-targets',
        type=int,
        default=150,
        help='Number of TESS targets to download (default: 150)'
    )
    
    # ZTF options
    parser.add_argument(
        '--start-date',
        type=str,
        default='2024-01-01',
        help='ZTF start date (YYYY-MM-DD, default: 2024-01-01)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2024-01-31',
        help='ZTF end date (YYYY-MM-DD, default: 2024-01-31)'
    )
    parser.add_argument(
        '--ztf-max-images',
        type=int,
        default=100,
        help='Maximum ZTF images to download (default: 100)'
    )
    parser.add_argument(
        '--ztf-fields',
        type=int,
        nargs='+',
        help='ZTF field IDs (space-separated)'
    )
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DataDownloadOrchestrator(
        base_dir=args.base_dir,
        config_path=args.config
    )
    
    # Check if at least one dataset was selected
    if not (args.all or args.kepler or args.tess or args.ztf):
        parser.error("Must specify --all or at least one of --kepler, --tess, --ztf")
    
    # Execute downloads
    if args.all:
        results = orchestrator.download_all(
            kepler_targets=args.kepler_targets,
            tess_targets=args.tess_targets,
            ztf_start_date=args.start_date,
            ztf_end_date=args.end_date,
            ztf_max_images=args.ztf_max_images
        )
    else:
        results = {}
        
        if args.kepler:
            results['kepler'] = orchestrator.download_kepler_data(
                max_targets=args.kepler_targets,
                mission=args.kepler_mission
            )
        
        if args.tess:
            results['tess'] = orchestrator.download_tess_data(
                max_targets=args.tess_targets
            )
        
        if args.ztf:
            results['ztf'] = orchestrator.download_ztf_data(
                start_date=args.start_date,
                end_date=args.end_date,
                field_ids=args.ztf_fields,
                max_images=args.ztf_max_images
            )
        
        orchestrator._print_summary(results)
    
    # Exit with appropriate code
    summary = results.get('summary', results)
    if any('error' in v for v in results.values() if isinstance(v, dict)):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()