# ComputerVisionTutorials.jl
[![Glass Notebook](https://img.shields.io/badge/Docs-Glass%20Notebook-aquamarine.svg)](https://glassnotebook.io/r/zQzph_NEw8WtbYAquAQG3/index.jl)

![ComputerVisionTutorials.jl Screenshot](/assets/image-seg-green.jpeg)

## Introduction
ComputerVisionTutorials.jl is a Julia package focused on deep learning for computer vision. This package is designed to provide a hands-on approach to learning state-of-the-art vision models and techniques through interactive Pluto notebooks. This package takes full advantage of the Glass Notebook system to provide meta-documentation. This means that each tutorial is not only a standalone guide but also part of a larger, interconnected web of resources. This allows for a more holistic and interconnected learning experience.

*Note: This package integrates several Git submodules like Losers.jl, DistanceTransforms.jl, and ComputerVisionMetrics.jl, all documented within the Glass Notebook system, ensuring seamless integration into the larger documentation framework.*

## Getting Started
To begin exploring deep learning for computer vision with Julia, [click this link](https://glassnotebook.io/r/zQzph_NEw8WtbYAquAQG3/index.jl). You can also clone this repo and navigate to the `tutorials/` directory, where you'll find Pluto notebooks that will guide you through various concepts and applications.

## Repository Structure
- ✅ (Complete)
- 🚧 (Work in Progress)
- 🔜 (Todo)

### Core Packages
The following Git submodules are integrated into this package, each offering unique functionalities vital to the computer vision pipeline. These packages are documented and interconnected within the Glass Notebook system, enhancing the learning experience.

- **Losers.jl**: A library for loss functions specific to deep learning in computer vision.
- **DistanceTransforms.jl**: Provides algorithms for computing distance transforms, essential in various computer vision tasks.
- **ComputerVisionMetrics.jl**: A collection of metrics for evaluating computer vision models.

You can clone ComputerVisionTutorials.jl repository and these submodules above at once with the --recursive option:

```bash
$ git clone --recursive https://github.com/Dale-Black/ComputerVisionTutorials.jl.git
```

You can also get submodules via `git submodule update --init` after ComputerVisionTutorials.jl is installed.

```bash
$ git clone https://github.com/Dale-Black/ComputerVisionTutorials.jl.git
$ cd ComputerVisionTutorials.jl
$ git submodule update --init
Submodule 'ComputerVisionMetrics.jl' (https://github.com/Dale-Black/ComputerVisionMetrics.jl) registered for path 'ComputerVisionMetrics.jl'
Submodule 'DistanceTransforms.jl' (https://github.com/Dale-Black/DistanceTransforms.jl) registered for path 'DistanceTransforms.jl'
Submodule 'Losers.jl' (https://github.com/Dale-Black/Losers.jl) registered for path 'Losers.jl'
Cloning into '.../ComputerVisionTutorials.jl/ComputerVisionMetrics.jl'...
Cloning into '.../ComputerVisionTutorials.jl/DistanceTransforms.jl'...
Cloning into '.../ComputerVisionTutorials.jl/Losers.jl'...
Submodule path 'ComputerVisionMetrics.jl': checked out 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
Submodule path 'DistanceTransforms.jl': checked out 'yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy'
Submodule path 'Losers.jl': checked out 'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz'
```

### Comprehensive Tutorials
Notebooks providing end-to-end tutorials on common computer vision tasks.

- 🔜 **Image Classification** (`tutorials/01_image_classification.jl`)
  - ...
- 🔜 **Object Tracking** (`tutorials/02_object_detection.jl`)
  - ...
- 🚧 **Image Segmentation** (`tutorials/03_image_segmentation.jl`)
  - Guide on 3D heart segmentation in CT images.
- 🔜 **Object Tracking** (`tutorials/02_object_detection.jl`)
  - ...

### Components
Notebooks focusing on core components of a deep learning pipeline.

- 🚧 **1. Data Preparation** (`components/01_data_preparation.jl`)
  - Techniques for data loading, augmentation, and dataset splitting.
- 🚧 **2. Model Building** (`components/02_model_building.jl`)
  - Exploration of layer architecture and activation functions.
- 🔜 **3. Training and Validation** (`components/03_training_validation.jl`)
  - Overview of optimization algorithms, regularization techniques, and validation strategies. Detailed look into different loss functions and their applications. Includes a section on model checkpointing, discussing strategies for saving and reloading model states during training.
- 🔜 **4. Model Evaluation** (`components/04_model_evaluation.jl`)
  - Discussion on performance metrics like accuracy, precision, recall, F1-score, ROC-AUC. Methods for interpreting confusion matrices and conducting error analysis.
- 🔜 **5. Model Deployment and Inference** (`components/05_model_deployment_inference.jl`)
  - Guides on exporting models, integrating with applications, and efficient model inference techniques.

## Contribution
We're open to contributions! Feel free to start the coming-soon notebooks, help out with the notebooks that are still in progress, or suggest and create new ones. We accept both pull requests and issues. Our core packages, like Losers.jl and ComputerVisionMetrics.jl, could also use many updates and additions. If you've got a Julia package that's useful for deep learning in computer vision, consider using [Glass Notebook](https://glassnotebook.io/) for your packages documentation and we'd be happy to integrate it directly into ComputerVisionTutorials.jl!
