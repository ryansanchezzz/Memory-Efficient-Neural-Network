Memory-Efficient CNN with Outer-Product Layers

Overview
This project implements a memory-efficient convolutional neural network (CNN) for the Fashion MNIST dataset. 
Instead of standard fully connected (FC) layers, it introduces Outer-Product Linear layers to drastically reduce memory usage while maintaining strong predictive performance.

The work was inspired by my coursework in Foundations of Machine Learning (Los Angeles, CA, Aug 2025 – Dec 2025).

Key Highlights:
* Replaced FC layers with Outer-Product representations, reducing parameter count.
* Achieved 50–90% memory reduction in fully connected layers with minimal accuracy loss.
* Evaluated trade-offs between model size, memory footprint, and predictive performance.
* End-to-end pipeline: data loading → model training → evaluation → per-class accuracy analysis.

Dependencies:
pip install -r requirements.txt
