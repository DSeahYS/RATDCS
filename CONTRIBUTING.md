# Contributing to RATDCS

Thank you for your interest in contributing to the Real-Time Asteroid Threat Detection & Collision-Avoidance System! This document provides guidelines and best practices for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Documentation](#documentation)
7. [Pull Request Process](#pull-request-process)
8. [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of experience level, background, or identity.

### Expected Behavior

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Docker and Docker Compose
- NVIDIA GPU (recommended for ML development)
- Familiarity with ML frameworks (TensorFlow/PyTorch)
- Basic understanding of orbital mechanics (helpful but not required)

### Setting Up Development Environment

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/ratdcs.git
cd ratdcs
```

2. **Create a development branch**

```bash
git checkout -b feature/your-feature-name
```

3. **Set up Python environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

4. **Install pre-commit hooks**

```bash
pre-commit install
```

5. **Configure environment**

```bash
cp .env.example .env
# Edit .env with your local settings
```

## Development Workflow

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Urgent production fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring
- `test/description` - Test additions/modifications

### Commit Message Format

Follow conventional commits specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Example:**

```
feat(detection): add multi-scale asteroid detection

Implement pyramid-based detection for improved small object detection.
Uses feature pyramid network (FPN) architecture.

Closes #123
```

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

- **Line length**: 100 characters (not 79)
- **Imports**: Use `isort` for automatic sorting
- **Formatting**: Use `black` for consistent formatting
- **Type hints**: Required for all public functions
- **Docstrings**: Google style docstrings

### Code Formatting

```bash
# Format code
black src/ tests/
isort src/ tests/

# Check formatting
black --check src/ tests/
isort --check-only src/ tests/
```

### Linting

```bash
# Run linters
pylint src/
flake8 src/
mypy src/
```

### Example Code Style

```python
"""Module for asteroid detection using CNN.

This module implements the asteroid detection pipeline based on
Van der Heijden et al. (2025) methodology.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import tensorflow as tf


class AsteroidDetector:
    """CNN-based asteroid detector for telescope imagery.
    
    Attributes:
        model_path: Path to trained model weights
        confidence_threshold: Minimum confidence for detection
        batch_size: Batch size for inference
    """
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.85,
        batch_size: int = 32
    ) -> None:
        """Initialize asteroid detector.
        
        Args:
            model_path: Path to model file (.h5)
            confidence_threshold: Detection threshold [0, 1]
            batch_size: Inference batch size
            
        Raises:
            ValueError: If threshold not in valid range
            FileNotFoundError: If model file doesn't exist
        """
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("Threshold must be in range [0, 1]")
            
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self._model = self._load_model()
    
    def detect(
        self,
        image: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Detect asteroids in telescope image.
        
        Args:
            image: Input image array (H, W, C)
            metadata: Optional metadata (timestamp, coordinates)
            
        Returns:
            List of detection dictionaries with bounding boxes,
            confidence scores, and estimated parameters
            
        Example:
            >>> detector = AsteroidDetector("models/cnn_v1.h5")
            >>> detections = detector.detect(image)
            >>> print(f"Found {len(detections)} asteroids")
        """
        # Implementation here
        pass
```

## Testing Guidelines

### Test Structure

- **Unit tests**: `tests/unit/` - Test individual components
- **Integration tests**: `tests/integration/` - Test component interactions
- **Validation tests**: `tests/validation/` - Validate ML model performance
- **Performance tests**: `tests/performance/` - Load and stress testing

### Writing Tests

```python
import pytest
import numpy as np
from src.detection.asteroid import AsteroidDetector


class TestAsteroidDetector:
    """Test suite for asteroid detector."""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing."""
        return AsteroidDetector(
            model_path="tests/fixtures/test_model.h5",
            confidence_threshold=0.85
        )
    
    def test_detection_output_format(self, detector):
        """Verify detection output structure."""
        image = np.random.rand(256, 256, 3).astype(np.float32)
        result = detector.detect(image)
        
        assert "detections" in result
        assert "processing_time_ms" in result
        assert isinstance(result["detections"], list)
    
    @pytest.mark.parametrize("threshold", [0.5, 0.75, 0.9])
    def test_confidence_threshold(self, threshold):
        """Test detector with different thresholds."""
        detector = AsteroidDetector(
            model_path="tests/fixtures/test_model.h5",
            confidence_threshold=threshold
        )
        assert detector.confidence_threshold == threshold
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test category
pytest tests/unit
pytest tests/integration

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_asteroid_detector.py -v

# Run tests matching pattern
pytest -k "asteroid"
```

### Test Coverage Requirements

- Minimum 85% code coverage for new code
- 100% coverage for critical components (control systems, safety checks)
- All public APIs must have tests
- Edge cases and error conditions must be tested

## Documentation

### Code Documentation

- All public functions/classes must have docstrings
- Use Google-style docstrings
- Include examples in docstrings where helpful
- Keep docstrings up-to-date with code changes

### API Documentation

- Document all REST endpoints
- Include request/response examples
- Document error codes and messages
- Update OpenAPI schema when changing APIs

### Architecture Documentation

- Update `ARCHITECTURE.md` for significant design changes
- Document design decisions and trade-offs
- Include diagrams for complex subsystems

## Pull Request Process

### Before Submitting

1. **Update from main branch**

```bash
git fetch upstream
git rebase upstream/main
```

2. **Run tests locally**

```bash
pytest
black --check src/ tests/
mypy src/
```

3. **Update documentation**

- Update README if adding features
- Add/update docstrings
- Update CHANGELOG.md

4. **Write clear commit messages**

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added/updated
```

### Review Process

1. PR must pass all CI checks
2. Requires approval from at least one maintainer
3. All review comments must be addressed
4. Squash commits before merging (if requested)

## Issue Reporting

### Bug Reports

Include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, etc.)
- Relevant logs/screenshots
- Minimal code example (if applicable)

### Feature Requests

Include:
- Clear description of proposed feature
- Use case and motivation
- Proposed implementation approach (optional)
- Willingness to contribute implementation

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature request
- `documentation`: Documentation improvements
- `good-first-issue`: Good for newcomers
- `help-wanted`: Extra attention needed
- `question`: Further information requested

## Development Tips

### Performance Profiling

```bash
# Profile code
python -m cProfile -o profile.stats script.py

# View results
python -m pstats profile.stats
```

### Debugging

```bash
# Run with debugger
python -m pdb script.py

# Or use ipdb
pip install ipdb
import ipdb; ipdb.set_trace()
```

### ML Model Development

- Use version control for model weights (DVC or similar)
- Log hyperparameters with MLflow/Weights & Biases
- Document model architecture and training process
- Include evaluation metrics and baselines
- Test on multiple scenarios/datasets

## Questions?

For questions or clarifications:

- Open a discussion on GitHub
- Join our Discord server
- Email: ratdcs-support@example.com

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project website (coming soon)

Thank you for contributing to RATDCS! 🚀