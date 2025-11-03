# RATDCS - Real-Time Asteroid Threat Detection & Collision-Avoidance System

[![CI Pipeline](https://github.com/organization/ratdcs/workflows/CI/badge.svg)](https://github.com/organization/ratdcs/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue.svg)](https://kubernetes.io/)

> **Production-grade, real-time space safety system for autonomous threat detection and collision avoidance in orbital environments.**

## 🌌 Overview

RATDCS integrates advanced machine learning with classical orbital mechanics to provide comprehensive situational awareness and autonomous response capabilities for spacecraft operations. The system processes multiple data streams in real-time to detect asteroids, classify exoplanets, predict space debris collisions, and execute autonomous avoidance maneuvers.

### Key Features

- **🛰️ Multi-Modal Threat Detection**: Asteroids, NEOs, exoplanets, and space debris
- **🤖 Autonomous Decision-Making**: RL-based attitude control with PPO
- **⚡ Real-Time Processing**: <500ms end-to-end latency
- **🎯 High Accuracy**: ≥90% precision for asteroid detection
- **🔧 Fault-Tolerant Control**: Mission-critical reliability with graceful degradation
- **📊 Production-Ready**: Microservices architecture with Kubernetes orchestration

## 📋 Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Development](#development)
- [Testing](#testing)
- [Deployment](#deployment)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## 🏗️ Architecture

For detailed architecture specifications, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
┌─────────────────────────────────────────────────────────────┐
│                    RATDCS System Architecture                │
└─────────────────────────────────────────────────────────────┘

[Data Sources] → [Kafka/RabbitMQ] → [Detection Pipeline]
                                          ↓
                              [Threat Assessment & Fusion]
                                          ↓
                              [Autonomous Decision System]
                                          ↓
                              [Fault-Tolerant Control]
                                          ↓
                              [Actuator Commands]
```

### Technology Stack

- **ML Frameworks**: TensorFlow 2.15, PyTorch 2.1, LightGBM 4.1, Stable-Baselines3 2.2
- **Scientific Computing**: NumPy, SciPy, Astropy, Poliastro
- **Backend**: FastAPI, gRPC, Kafka, RabbitMQ
- **Databases**: PostgreSQL, Redis, InfluxDB
- **Orchestration**: Kubernetes, Docker, Helm
- **Monitoring**: Prometheus, Grafana, Jaeger

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ (recommended: 3.10.12)
- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.2+ (for training/inference)
- 16GB+ RAM
- Git

### Local Development Setup

1. **Clone the repository**

```bash
git clone https://github.com/organization/ratdcs.git
cd ratdcs
```

2. **Set up Python environment**

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

3. **Configure environment variables**

```bash
cp .env.example .env
# Edit .env with your settings
```

4. **Start services with Docker Compose**

```bash
docker-compose up -d
```

5. **Verify installation**

```bash
# Check service health
docker-compose ps

# Run tests
pytest tests/unit -v
```

6. **Access services**

- API: http://localhost:8000
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- Jaeger UI: http://localhost:16686

## 📦 Installation

### From Source

```bash
# Install package in development mode
pip install -e .

# Or install specific components
pip install -e ".[gpu]"  # With GPU support
pip install -e ".[dev]"  # With development tools
```

### Using Docker

```bash
# Build all services
docker-compose build

# Build specific service
docker build -t ratdcs-asteroid-detector -f docker/Dockerfile.asteroid-detector .
```

## ⚙️ Configuration

RATDCS uses hierarchical YAML configuration:

- **`config/default.yaml`**: Base configuration
- **`config/development.yaml`**: Development overrides
- **`config/production.yaml`**: Production settings

Example configuration:

```yaml
detection:
  asteroid:
    model_path: "models/asteroid_detector/cnn_v1.h5"
    confidence_threshold: 0.85
    batch_size: 32

decision:
  rl_controller:
    policy_path: "models/rl_controller/ppo_policy_v1.zip"
    action_frequency_hz: 20
```

See [`.env.example`](.env.example) for environment variables.

## 💻 Usage

### Running Detection Services

```bash
# Asteroid detection
python -m src.detection.asteroid.detector

# Exoplanet classification
python -m src.detection.exoplanet.classifier

# Debris collision prediction
python -m src.detection.debris.predictor
```

### Running Decision Services

```bash
# RL-based controller
python -m src.decision.rl_controller.ppo_agent

# Fault-tolerant control
python -m src.decision.fault_tolerant.control_allocator
```

### API Server

```bash
# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Access API docs
open http://localhost:8000/docs
```

### Using the API

```python
import requests

# Detect asteroids
response = requests.post(
    "http://localhost:8000/api/v1/detect/asteroid",
    files={"image": open("telescope_image.fits", "rb")}
)

detections = response.json()
print(f"Found {len(detections['detections'])} asteroids")
```

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint
pylint src/
flake8 src/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## 🧪 Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit

# With coverage
pytest tests/unit --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_asteroid_detector.py -v
```

### Integration Tests

```bash
# Run integration tests (requires services)
pytest tests/integration

# With Docker Compose
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Performance Tests

```bash
# Run load tests
locust -f tests/performance/test_load.py --host=http://localhost:8000
```

## 🚢 Deployment

### Kubernetes Deployment

```bash
# Deploy to Kubernetes cluster
kubectl apply -f kubernetes/namespaces/
kubectl apply -f kubernetes/deployments/
kubectl apply -f kubernetes/services/

# Check deployment status
kubectl get pods -n ratdcs-detection
kubectl get pods -n ratdcs-decision

# View logs
kubectl logs -f deployment/asteroid-detector -n ratdcs-detection
```

### Using Helm (Coming Soon)

```bash
helm install ratdcs ./helm/ratdcs \
  --namespace ratdcs \
  --create-namespace \
  --values values-production.yaml
```

## 📚 API Documentation

### REST API Endpoints

- `POST /api/v1/detect/asteroid` - Detect asteroids in telescope images
- `POST /api/v1/classify/exoplanet` - Classify exoplanet candidates
- `POST /api/v1/predict/debris` - Predict debris collision risk
- `POST /api/v1/control/maneuver` - Execute avoidance maneuver
- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

### gRPC Services

- `ThreatDetectionService` - Detection pipeline
- `DecisionService` - Autonomous control
- `TelemetryService` - System monitoring

Full API documentation available at `/docs` when server is running.

## 📊 Performance Targets

| Component | Latency Target | Accuracy Target |
|-----------|---------------|-----------------|
| Asteroid Detector | <200ms | ≥90% precision |
| Exoplanet Classifier | <300ms | >0.94 AUC |
| Debris Predictor | <400ms | <0.10 ECE |
| RL Controller | <50ms | ≥95% success |
| End-to-End Pipeline | <500ms | Mission-critical |

## 🤝 Contributing

We welcome contributions! Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [`LICENSE`](LICENSE) file for details.

## 👨‍🚀 Author

**Dave Seah Yong Sheng**
NTU Aerospace Engineering
Nanyang Technological University, Singapore

- **GitHub**: https://github.com/DSeahYS/RATDCS
- **Email**: daveseahys@gmail.com

##  Acknowledgments

- **Van der Heijden et al. (2025)** - CNN methodology for asteroid detection
- **Malik et al. (2022)** - TSFresh feature extraction methodology achieving 0.948 AUC
- **Zwicky Transient Facility (ZTF)** - Survey data
- **TESS Mission** - Exoplanet light curves
- **Space-Track.org** - Orbital debris catalog

## 📞 Contact & Support

- **Project Lead**: Dave Seah Yong Sheng
- **GitHub Repository**: https://github.com/DSeahYS/RATDCS
- **Issues**: https://github.com/DSeahYS/RATDCS/issues

## 🎯 Roadmap

- [x] Core detection pipeline
- [x] RL-based controller
- [x] Fault-tolerant control
- [ ] Advanced orbit propagation
- [ ] Multi-satellite coordination
- [ ] Real-time data fusion improvements
- [ ] Enhanced ML model compression
- [ ] Edge deployment optimization

---

**Built with ❤️ for space safety and exploration**