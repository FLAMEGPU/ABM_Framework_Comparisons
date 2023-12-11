# pyFLAMEGPU ABM Comparison Benchmarks

This directory contains the implementation of several ABMs used to compare agent based models, including:

+ `boids2D` - a 2D spatial implementation of boids flocking model
+ `schelling` - an implementation of Schelling's Model of Segregation

The [`pyflamegpu` package](https://github.com/FLAMEGPU/FLAMEGPU2#installation) must be available within the Python environment used to launch the independent models or `benchmark.py`. The upto-date requirements for running pyFLAMEGPU can be found within the same document, however typically a recent CUDA capable GPU, CUDA toolkit and Python 3 are the core requirements.

For details on how to develop a model using pyFLAMEGPU (or FLAME GPU 2) or the requirements for using pyFLAMEGPU, refer to the [userguide & API documentation](https://docs.flamegpu.com/).

