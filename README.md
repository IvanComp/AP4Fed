---
Version: [Alpha-1.0.0]
Main Libraries: [Flower, PyTorch, PyTorchGAN, Torchvision]
Testing Datasets: [CIFAR-10, FMNIST, NYUv2, iv4N]
Testing Models: [CNN, U-Net, DenseNet]
---

# AP4FED

<p align="center">
<img src="img/readme/logo.svg" width="340px" height="210px"/>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/IvanComp/AP4FED">
  <img src="https://img.shields.io/badge/python-3.12+-yellow">
  <img src="https://img.shields.io/badge/docker-3.10+-blue">
  <img src="https://img.shields.io/badge/docker%20compose-2.29+-blue">
  <img src="https://img.shields.io/badge/Tool%20v.-1.0-green" alt="Version">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open to Colab">
</p>


<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-ee4b2b?logo=numpy&logoColor=fff">  
  <img src="https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff"> 
  <img src="https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff">
</p>

<tt>AP4FED</tt> is a Federated Learning Benchmark platform built on top of the [Flower](https://github.com/adap/flower) an open-source Python library that simplifies building Federated Learning systems. It enables the definition of customized Federated Learning system configurations by leveraging architectural patterns, aiming to extract and analyze system performance metrics.

<tt>AP4FED</tt> supports software architects by offering a framework for the quantitative evaluation of specific design decisions, enabling a deeper understanding of their impact on system performance and guiding the optimization of Federated Learning architectures.

An initial version of <tt>AP4FED</tt> was developed and tested in the research paper "_Performance Analysis of Architectural Patterns for Federated Learning Systems_" accepted for the 22nd IEEE International Conference on Software Architecture (ICSA 2025) [1].

<p align="center">
<img src="img/readme/Poster.png" width="340px" height="210px"/>
</p>

# Table of contents
<!--ts-->
   * [Prerequisites](#prerequisites)
   * [How to Run](#how-to-run)
   * [Framework](#framework)
   * [Architectural Patterns](#architectural-patterns)
   * [References](#references)
   
# Prerequisites

To run <tt>AP4FED</tt>, ensure that the following prerequisites are met:

- [Python (version 3.12.X or higher)](https://www.python.org/downloads/) 
- [Docker](https://docs.docker.com/get-docker/)

Docker (Docker Compose) is not required to run local Federated Learning projects, but they are valuable tools for configurations that emulate real clients, particularly by distributing physical resources such as CPUs while avoiding CPU overcommitment. The framework allows flexibility between running a fully containerized setup or opting for a local simulation, depending on the project’s requirements and the desired level of fidelity in emulating a distributed system.

- **Docker**: Required to create and run containers.
- **Docker Compose**: Enables running multi-container applications with Docker using the `docker-compose.yml` file.

You can verify the installation with the following commands:
```bash
docker --version
docker compose version
```

# How To Run

Please ensure that all [Prerequisites](#Prerequisites) are met before proceeding with the installation.
In the main folder run the following command:

```bash
pip install -r requirements.txt && python home.py
```

It will install all the required Python libraries. 

Then, follow the instructions on the GUI of <tt>AP4FED</tt> to configure and run a FL simulation. This interface allows you to configure the Federated Learning system and proceed with benchmarking, providing a user-friendly way to manage and test the setup (Please see the next [Section](#Framework))

# Framework

## FL Simulation Process General Overview

<p align="center">
<img src="img/readme/AP4FED_process.png" width="850px" height="250px"/>
</p>

Users can choose to create a new project or load an existing JSON configuration to define system parameters, such as dataset properties, client selection criteria, and training mechanisms. For example, the Heterogeneous Data Handler allows synthetic data generation to augment client datasets and address non-IID data distributions. After configuration, the simulation process orchestrates interactions between clients and the server, tracks system metrics such as training, communication, and round times, and outputs a detailed report. This enables users to test design decisions and optimize FL system performance with ease.

## AP4FED GUI

<p align="center">
<img src="img/readme/1.png" width="400px" height="300px"/>
<img src="img/readme/2.png" width="400px" height="300px"/>
</p>

<p align="center">
<img src="img/readme/3.png" width="400px" height="300px"/>
<img src="img/readme/4.png" width="400px" height="300px"/>
</p>

<p align="center">
<img src="img/readme/5.png" width="400px" height="300px"/>
<img src="img/readme/6.png" width="400px" height="300px"/>
</p>

<p align="center">
<img src="img/readme/7.png" width="400px" height="300px"/>
<img src="img/readme/8.png" width="400px" height="300px"/>
</p>


## Load/Save a Custom Configuration

<p align="center">
<img src="img/readme/7.png" width="180px" height="280px"/>
</p>

## Model Parameters

| Parameter | Model n/2 | Model n | Model n*2 |  Model ImageNet100
| --- | --- | --- | --- | --- |
| `Conv. 1` | 3 filters, 5x5 kernel | 6 filters, 5x5 kernel | 12 filters, 5x5 kernel | --- |
| `Pool` | Max pooling, 2x2 kernel | Max pooling, 2x2 kernel | Max pooling, 2x2 kernel | --- |
| `Conv. 2` | 8 filters, 5x5 kernel | 16 filters, 5x5 kernel | 32 filters, 5x5 kernel | --- |
| `FC 1` | 60 units | 120 units | 240 units | --- |
| `FC 2` | 42 units | 84 units | 168 units | --- |
| `FC 3` | 10 units | 20 units | 30 units | --- |
| `Batch Size` | 32 | 32 | 32 | --- |
| `Learning Rate` | 0.001 | 0.001 | 0.001 | --- |
| `Optimizer` | SGD | SGD | SGD | --- |

<p align="center">
<img src="img/readme/7.png" width="180px" height="280px"/>
</p>

# Architectural Patterns

The Architectural Patterns implemented in our framework are:

| Architectural Pattern | Pattern Category | Description |
| --- | --- | --- | 
| **Client Registry** | `Client Management` | A registry to store relevant information of each client device participating in Federated Learning rounds. | 
| **Client Selector** | `Client Management` | A mechanism to dynamically select clients based on specific criteria, such as data distribution, device capabilities, or network conditions, to optimize the federated learning process. | 
| **Client Cluster** | `Client Management` | A strategy to group clients into clusters based on shared characteristics, such as data similarity or device attributes, improving model accuracy and convergence in non-IID data scenarios. | 
| **Message Compressor** | `Model Management` | A component designed to reduce the size of data exchanged between clients and the server by compressing messages, which lowers communication latency and bandwidth usage in federated learning environments. |
| **Multi-Task Model Trainer** | `Model Training` | Allow to train different global models simultaneously |
| **Heterogeneous Data Handler** | `Model Training` | Enables pre-processing operations to enhance the quality of datasets for each client participating in the FL process. |

The **Client Registry** architectural pattern is implemented by adding the following parameters:

| Attribute | Data Type | Description |
| --- | --- | --- | 
| **cID** | `string` | Client’s Unique Identifier | 
| **cluster_Type** | `string` | Cluster associated to each Client | 
| **n_CPU** | `int` | Number of Client’s CPU | 
| **training_time** | `float` | Client’s Training Time | 
| **communication_time** | `float` | Client’s Communication Time | 
| **total_round_time** | `float` | Client’s Total Round Time | 

You can explore use cases and performance testing of architectural patterns [1] using <tt>AP4FED</tt> in the following Zenodo repository: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14539962.svg)](https://zenodo.org/records/14539962)

# References

[1] Compagnucci, I., Pinciroli, R., & Trubiani, C. (2025). **Performance Analysis of Architectural Patterns for Federated Learning Systems.**
Accepted for the 22nd IEEE International Conference on Software Architecture. ICSA 2025.

Check out the paper _[Performance Analysis of Architectural Patterns for Federated Learning Systems](), ICSA 2025_, and please cite it if you use <tt>AP4FED</tt> in your work.

```
@inproceedings{CompagnucciFL,
  author       = {Compagnucci, Ivan and Pinciroli, Riccardo and Trubiani, Catia},
  title        = {{Performance Analysis of Architectural Patterns for Federated Learning Systems}},
  booktitle    = {International Conference on Software Architecture, {ICSA 25}},
  pages        = {},
  publisher    = {{IEEE}},
  year         = {2025},
}
```

