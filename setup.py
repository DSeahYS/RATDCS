"""
RATDCS Setup Configuration
Real-Time Asteroid Threat Detection & Collision-Avoidance System
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
requirements = [req.strip() for req in requirements if req.strip() and not req.startswith("#")]

dev_requirements = (this_directory / "requirements-dev.txt").read_text().splitlines()
dev_requirements = [req.strip() for req in dev_requirements if req.strip() and not req.startswith("#")]

setup(
    name="ratdcs",
    version="1.0.0",
    author="RATDCS Team",
    author_email="ratdcs-support@example.com",
    description="Real-Time Asteroid Threat Detection & Collision-Avoidance System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/organization/ratdcs",
    project_urls={
        "Documentation": "https://docs.ratdcs.org",
        "Bug Tracker": "https://github.com/organization/ratdcs/issues",
        "Source Code": "https://github.com/organization/ratdcs",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "gpu": [
            "tensorflow-gpu==2.15.0",
            "torch-cuda==2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ratdcs-api=api.main:main",
            "ratdcs-detector=detection.asteroid.detector:main",
            "ratdcs-classifier=detection.exoplanet.classifier:main",
            "ratdcs-predictor=detection.debris.predictor:main",
            "ratdcs-controller=decision.rl_controller.ppo_agent:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)