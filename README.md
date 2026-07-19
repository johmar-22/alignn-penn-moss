# Multi-Task ALIGNN Prediction of Refractive Index and Bandgap: Quantifying Transfer, Joint Conformal Bounds, and the Penn-Moss Tradeoff




**Author:** Johaimen M. Omar  
**Affiliations:** Kastamonu University, Türkiye; Mindanao State University - Marawi Campus, Philippines  **Journal:** Computational Materials Science (Submitted)

## Overview

This repository hosts the official source code, calibrated conformal models, and virtual screening matrices for evaluating the Atomistic Line Graph Neural Network (ALIGNN) within a multi-task learning framework. 

By mapping the joint distribution of optical refractive index ($n$) and electronic bandgap ($E_g$) across 8,038 inorganic materials from a pinned JARVIS-DFT snapshot, we analyze the limits of structural representational complementarity under rigorous out-of-distribution (OOD) crystal-prototype splits. Additionally, we provide the joint split-conformal prediction intervals and the virtual screening workflow used to confirm the physical limitations imposed by the Penn-Moss tradeoff relationship.

## System Specifications

| Requirement | Minimum Configuration | Verified Operational Environment |
| :--- | :--- | :--- |
| **Python** | 3.9 | 3.10 |
| **RAM** | 16 GB | 32 GB |
| **GPU** | Optional (CPU Inference Supported) | NVIDIA RTX 4070 / CUDA 12.1 |
| **Disk Storage** | 1 GB Free Space | 5 GB Free Space |
| **Operating System** | Windows 10/11 / macOS | Ubuntu 22.04 LTS |

*Estimated Inference Runtime:* < 5 minutes on standard desktop configurations using a pre-computed graph setup.

## Installation

Clone the repository environment locally and install the dependencies into a clean virtual environment:

```bash
git clone https://github.com/johmar-22/alignn-penn-moss.git
cd alignn-penn-moss
pip install -r requirements.txt
