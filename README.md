# ComputerVisionTutorials.jl
[![Glass Notebook](https://img.shields.io/badge/Docs-Glass%20Notebook-aquamarine.svg)](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl)

![ComputerVisionTutorials.jl Screenshot](/assets/screenshot.jpeg)

## Introduction
ComputerVisionTutorials.jl is a Julia package focused on deep learning for computer vision. This package is designed to provide a hands-on approach to learning state-of-the-art vision models and techniques through interactive Pluto notebooks. This package takes full advantage of the Glass Notebook system to provide meta-documentation. This means that each tutorial is not only a standalone guide but also part of a larger, interconnected web of resources. This allows for a more holistic and interconnected learning experience.

*Note: This package integrates several Git submodules like Losers.jl, DistanceTransforms.jl, and ComputerVisionMetrics.jl, all documented within the Glass Notebook system, ensuring seamless integration into the larger documentation framework.*

## Repository Structure

### Core Packages
The following Git submodules are integrated into this package, each offering unique functionalities vital to the computer vision pipeline. These packages are documented and interconnected within the Glass Notebook system, enhancing the learning experience.

- **Losers.jl**: A library for loss functions specific to deep learning in computer vision.
- **DistanceTransforms.jl**: Provides algorithms for computing distance transforms, essential in various computer vision tasks.
- **ComputerVisionMetrics.jl**: A collection of metrics for evaluating computer vision models.

### Components
Notebooks focusing on core components of a deep learning pipeline.

- **Data Preparation** (`components/01_data_preparation.jl`)
  - Content: Techniques for data loading, augmentation, and dataset splitting.
- **Model Building** (`components/02_model_building.jl`)
  - Content: Exploration of layer architecture and activation functions.
- **Training and Validation** (`components/03_training_validation.jl`)
  - Content: Overview of optimization algorithms, regularization techniques, and validation strategies. Detailed look into different loss functions and their applications.
- **Model Evaluation** (`components/04_model_evaluation.jl`)
  - Content: Discussion on performance metrics like accuracy, precision, recall, F1-score, ROC-AUC. Methods for interpreting confusion matrices and conducting error analysis.
- **Model Deployment and Inference** (`components/05_model_deployment_inference.jl`)
  - Content: Guides on exporting models, integrating with applications, and efficient model inference techniques.

### Comprehensive Tutorials
Notebooks providing end-to-end tutorials on common computer vision tasks.

- **Image Classification** (`tutorials/01_image_classification.jl`)
  - Content: Building a classifier, improving accuracy, and reducing overfitting.
- **Object Detection** (`tutorials/02_object_detection.jl`)
  - Content: Implementing models like YOLO, practical tips for bounding box annotations.
- **Image Segmentation** (`tutorials/03_image_segmentation.jl`)
  - Content: Guide on 3D heart segmentation in CT images.
- **Generative Models** (`tutorials/04_generative_models.jl`)
  - Content: Introduction to GANs and applications in image generation.
- **Pose Estimation** (`tutorials/05_pose_estimation.jl`)
  - Content: Building a pose estimation model, applications in sports analytics or motion capture.
- **Facial Recognition** (`tutorials/06_facial_recognition.jl`)
  - Content: Implementing a facial recognition system, discussion on privacy and ethical considerations.
- **Finetuning** (`tutorials/07_finetuning.jl`)
  - Content: Loading a pretrained model from Metalhead.jl/Boltz.jl and finetuning for a specific application.
- **Distributed Segmentation** (`tutorials/08_distributed_segmentation.jl`)
  - Content: Using FluxMPI.jl (or Dagger.jl?) for training a segmentation model accross multiple GPUs.

## Contribution
We welcome contributions! Feel free to help in completing the in-progress notebooks, or suggest and develop new ones. Pull requests and issues welcome!

## Getting Started
[![Glass Notebook](https://img.shields.io/badge/Docs-Glass%20Notebook-aquamarine.svg)](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl)

To begin exploring computer vision with Julia check out the links above. You can also clone this repo and navigate to the tutorials/ directory, where you'll find Pluto notebooks that will guide you through various concepts and applications.
