# ComputerVisionTutorials.jl
[![Glass Notebook](https://img.shields.io/badge/Docs-Glass%20Notebook-aquamarine.svg)](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl)

![ComputerVisionTutorials.jl Screenshot](/assets/screenshot.jpeg)


## Introduction
ComputerVisionTutorials.jl is a Julia package focused on deep learning for computer vision. This package is designed to provide a hands-on approach to learning state-of-the-art vision models and techniques through interactive Pluto notebooks. This package takes full advantage of the Glass Notebook system to provide meta-documentation. This means that each tutorial is not only a standalone guide but also part of a larger, interconnected web of resources. This allows for a more holistic and interconnected learning experience.

## Package Structure
```
ComputerVisionTutorials.jl/
│
├── components/                # Core components of deep learning
│   ├── 01_data_processing.jl     # (TODO) Data acquisition, cleaning, augmentation, annotation, etc
│   ├── 02_model_building.jl      # (TODO) CNN, RNN, GAN, Transformers, etc
│   ├── 03_optim_loss.jl          # (TODO) Loss functions, optimization techniques, etc
│   ├── 04_training.jl            # (TODO) Training metrics, logging, checkpointing, etc
│   └── 05_inference.jl           # (TODO) Model inference strategies, fine-tuning, etc
│
├── tutorials/                 # In-depth guides on specific topics
│   ├── 3D_segmentation.jl        # (IN PROGRESS)
│   ├── diffusion_models.jl       # (TODO)
│   ├── image_classification.jl   # (TODO)
│   └── distributed_training.jl   # (TODO)
│
└── README.md                  # Comprehensive guide to this package
```

*Note: This package integrates several Git submodules like Losers.jl, DistanceTransforms.jl, and ComputerVisionMetrics.jl, all documented within the Glass Notebook system, ensuring seamless integration into the larger documentation framework.*

## Contributing
We welcome contributions of all forms to help expand and improve this package. Whether you're interested in writing tutorials, enhancing documentation, or adding new features, your input is invaluable. Please don't hesitate to open up issues and submit pull requests.

## Getting Started
[![Glass Notebook](https://img.shields.io/badge/Docs-Glass%20Notebook-aquamarine.svg)](https://glassnotebook.io/r/DxnIPJnIqpEqiQnJgqiBP/index.jl)

To begin exploring computer vision with Julia check out the links above. You can also clone this repo and navigate to the tutorials/ directory, where you'll find Pluto notebooks that will guide you through various concepts and applications.
