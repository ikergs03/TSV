# TSV - Visual Signal Processing and Computer Vision Portfolio

This repository contains practical coursework for the Visual Signal Processing course (Tratamiento de Senales Visuales / Tratamiento de Senales Multimedia I) at EPS-UAM.

The work is organized as a progression from numerical signal processing fundamentals to classical computer vision and deep learning for scene recognition.

## What This Project Covers

- Foundational array and image operations with NumPy.
- Multi-scale image processing with Gaussian and Laplacian pyramids.
- Local feature detection and matching workflows.
- Scene classification with Bag of Words / Bag of Features pipelines.
- Deep learning experiments for scene recognition in Jupyter notebooks.

## Repository Structure

- `inicio/`: Environment setup and dependency list (`requirements.txt`).
- `p0/`: Introductory tasks on NumPy and basic matrix/image operations.
- `p1/`: Image fusion with pyramid-based reduce/expand and reconstruction.
- `p2/`: Local features and interest points (for example, Harris detector tasks).
- `p3/`: Scene recognition with classical ML (BoW/BoF + KMeans + classifiers).
- `p4/`: Notebook-based deep learning workflow (setup, datasets, network, training).

## Tech Stack

- Python
- NumPy, SciPy
- scikit-image, OpenCV, Pillow
- scikit-learn
- Matplotlib
- Jupyter Notebook
- TensorFlow (mainly in `p4/` notebooks)

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

	```bash
	pip install -r inicio/requirements.txt
	```

3. Run scripts from each practice folder as needed, for example:

	```bash
	python p1/p1_tarea1.py
	python p2/p2_tarea1.py
	python p3/p3_tarea1.py
	```

4. Open `p4/` notebooks in Jupyter or Google Colab for the deep learning part.

## Notes for Reproducibility

- Some scripts include local absolute paths from development machines; update those paths to match your environment.
- Some datasets may be expected in local folders used during the course.
- Comments and assignment context are partly in Spanish because this is original academic coursework.

## Portfolio Goal

This repository is published as a portfolio artifact to showcase practical computer vision and machine learning implementation skills across multiple levels of complexity.