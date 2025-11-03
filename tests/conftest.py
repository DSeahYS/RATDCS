"""
Shared pytest fixtures for RATDCS testing.

This module provides common test fixtures used across all test suites.
"""

import os
import sys
from pathlib import Path
from typing import Generator, Dict, Any
import tempfile

import pytest
import numpy as np
import yaml

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """Return test data directory path."""
    return project_root / "tests" / "fixtures"


@pytest.fixture(scope="session")
def config_dir(project_root: Path) -> Path:
    """Return config directory path."""
    return project_root / "config"


@pytest.fixture
def test_config(config_dir: Path) -> Dict[str, Any]:
    """Load test configuration."""
    config_path = config_dir / "development.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Mock Data Fixtures
# =============================================================================

@pytest.fixture
def mock_asteroid_image() -> np.ndarray:
    """Generate mock telescope image for asteroid detection."""
    # 256x256 RGB image with random noise
    image = np.random.rand(256, 256, 3).astype(np.float32)
    
    # Add synthetic asteroid (bright spot)
    center_x, center_y = 128, 128
    for i in range(center_x - 5, center_x + 5):
        for j in range(center_y - 5, center_y + 5):
            image[i, j] = [0.9, 0.9, 0.9]
    
    return image


@pytest.fixture
def mock_light_curve() -> Dict[str, np.ndarray]:
    """Generate mock light curve data for exoplanet classification."""
    time = np.linspace(0, 10, 1000)  # 10 day observation
    flux = np.ones_like(time) + np.random.normal(0, 0.001, time.shape)
    
    # Add synthetic transit
    transit_depth = 0.01
    transit_duration = 0.1
    transit_center = 5.0
    
    mask = np.abs(time - transit_center) < transit_duration / 2
    flux[mask] -= transit_depth
    
    return {
        'time': time.astype(np.float64),
        'flux': flux.astype(np.float32),
        'flux_err': np.full_like(flux, 0.001, dtype=np.float32)
    }


@pytest.fixture
def mock_debris_state() -> Dict[str, Any]:
    """Generate mock debris state for collision prediction."""
    return {
        'object_state': {
            'position': [6800.0, 1200.0, 500.0],  # ECI coordinates (km)
            'velocity': [7.5, -0.2, 0.1],  # km/s
            'covariance': np.eye(6).tolist(),
            'timestamp': '2025-11-02T14:00:00Z'
        },
        'debris_catalog': [
            {
                'norad_id': '12345',
                'tle_line1': '1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927',
                'tle_line2': '2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537',
                'rcs': 100.0,
                'mass': 420000.0
            }
        ],
        'environmental': {
            'solar_flux': 150.0,
            'geomagnetic_index': 2.5,
            'atmospheric_density': 1e-12
        }
    }


@pytest.fixture
def mock_spacecraft_state() -> Dict[str, Any]:
    """Generate mock spacecraft state for RL controller."""
    return {
        'attitude': {
            'quaternion': [1.0, 0.0, 0.0, 0.0],
            'angular_velocity': [0.0, 0.0, 0.0]
        },
        'position_velocity': {
            'position_eci': [7000.0, 0.0, 0.0],
            'velocity_eci': [0.0, 7.5, 0.0]
        },
        'threat_vectors': [],
        'constraints': {
            'fuel_remaining': 100.0,
            'power_available': 1000.0,
            'communication_window': True,
            'sun_angle': 45.0
        }
    }


# =============================================================================
# Model Fixtures
# =============================================================================

@pytest.fixture
def mock_cnn_model_path(temp_dir: Path) -> Path:
    """Create mock CNN model file."""
    model_path = temp_dir / "test_model.h5"
    # Create empty file as placeholder
    model_path.touch()
    return model_path


@pytest.fixture
def mock_lightgbm_model_path(temp_dir: Path) -> Path:
    """Create mock LightGBM model file."""
    model_path = temp_dir / "test_model.pkl"
    model_path.touch()
    return model_path


# =============================================================================
# Service Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def kafka_available() -> bool:
    """Check if Kafka is available for testing."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 9092))
        sock.close()
        return result == 0
    except:
        return False


@pytest.fixture(scope="session")
def redis_available() -> bool:
    """Check if Redis is available for testing."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(('localhost', 6379))
        sock.close()
        return result == 0
    except:
        return False


@pytest.fixture
def skip_if_no_kafka(kafka_available: bool):
    """Skip test if Kafka is not available."""
    if not kafka_available:
        pytest.skip("Kafka not available")


@pytest.fixture
def skip_if_no_redis(redis_available: bool):
    """Skip test if Redis is not available."""
    if not redis_available:
        pytest.skip("Redis not available")


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if GPU is not available."""
    try:
        import tensorflow as tf
        if not tf.config.list_physical_devices('GPU'):
            pytest.skip("GPU not available")
    except:
        pytest.skip("TensorFlow not available")


# =============================================================================
# Performance Testing Fixtures
# =============================================================================

@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Return performance thresholds for validation."""
    return {
        'asteroid_detector_latency_ms': 200,
        'exoplanet_classifier_latency_ms': 300,
        'debris_predictor_latency_ms': 400,
        'rl_controller_latency_ms': 50,
        'asteroid_precision': 0.90,
        'exoplanet_auc': 0.94,
        'rl_success_rate': 0.95
    }


@pytest.fixture
def benchmark_timer():
    """Provide timer for performance benchmarking."""
    import time
    
    class Timer:
        def __init__(self):
            self.start_time = None
            self.elapsed_ms = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            if self.start_time is None:
                raise RuntimeError("Timer not started")
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
            return self.elapsed_ms
    
    return Timer()


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def cleanup_test_artifacts(temp_dir: Path):
    """Automatically cleanup test artifacts after each test."""
    yield
    # Cleanup happens automatically with tempfile.TemporaryDirectory


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "validation" in str(item.fspath):
            item.add_marker(pytest.mark.validation)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)