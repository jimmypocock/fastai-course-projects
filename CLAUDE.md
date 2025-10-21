# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a collection of deep learning projects from the FastAI Practical Deep Learning course. The repository focuses on computer vision applications with real-world datasets, primarily using Google Colab for development.

## Development Environment

- **Primary Platform:** Google Colab (notebooks are designed to run with free GPU access)
- **Frameworks:** TensorFlow 2.x / PyTorch with FastAI
- **Target Python Version:** 3.x (compatible with Google Colab)

## Repository Structure

Projects are organized by topic in numbered directories:
- `01-image-recognition/` - Image classification projects (Sign Language MNIST, Fresh/Rotten Fruits)
- `utils/` - Shared helper functions and utilities
- `data/` - Local datasets (not tracked in git; notebooks download data as needed)
- `models/` - Saved model files (not tracked in git)
- `results/` - Performance metrics and visualizations (not tracked in git)

## Working with Notebooks

All project work is done in Jupyter notebooks (`.ipynb` files) designed for Google Colab:
- Each notebook is self-contained with its own setup and dependencies
- Notebooks include "Open in Colab" badges in README.md
- Data is downloaded within notebooks, not committed to the repository
- GPU runtime is recommended for training

## Project Structure

Current projects:
1. **Sign Language MNIST Classifier** - CNN for ASL recognition (24 classes, target >95% accuracy)
2. **Fresh vs Rotten Fruit Classifier** - Multi-class freshness detection (12 classes, target >92% accuracy)

Each project demonstrates specific techniques:
- Data augmentation and preprocessing
- Transfer learning (MobileNetV2, ResNet50)
- Model evaluation on unseen test sets
- Real-time prediction capabilities

## Key Practices

- Models trained on open datasets (TensorFlow Datasets, Kaggle)
- Focus on socially beneficial applications
- Detailed explanations and comments in notebooks
- Realistic performance metrics using test set evaluation
- All development uses free resources
