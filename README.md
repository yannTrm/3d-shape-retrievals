# 3D Shape Retrieval Project using View-Based Descriptors

This project focuses on 3D shape retrieval using View-Based Descriptors. The code provided here is designed to work with the McGill dataset, which can be accessed [here](https://www.cim.mcgill.ca/~shape/benchMark/).

## Overview

This project aims to implement and evaluate 3D shape retrieval techniques based on View-Based Descriptors. The McGill dataset serves as the benchmark for testing and comparing the performance of the implemented algorithms.

## Dataset

To get started, download the McGill dataset from [this link](https://www.cim.mcgill.ca/~shape/benchMark/). Ensure that you have the dataset stored in the appropriate directory as specified in the code.
## Usage

1. **Generate 2D Views:**
   First, run the `createFullDataset` notebook. This notebook is responsible for generating 2D views from our 3D models. Ensure that the necessary configurations are set correctly before running the notebook.

2. **Train Classification Model:**
   Next, proceed to the `classification` notebook. This notebook is designed to train a classification model, specifically using ResNet18, which will be used later as a feature extractor (latent vector). Make sure to follow the instructions in the notebook and adjust any parameters as needed.

3. **Create Latent Vectors:**
   Following the model training, proceed to the `createLatentVector` notebook. This notebook is responsible for saving the latent vectors of the 2D views of our reference 3D models into our database. Ensure that the necessary configurations are set correctly before running the notebook.

4. **3D Shape Retrieval:**
   Once the latent vectors are saved, proceed to the `retrieval` notebook. This notebook contains the application for performing 3D shape retrieval. Given a 3D shape, the application generates 2D views, extracts latent vectors, and compares them with the latent vectors of our reference data. It searches for the closest match based on the minimum distance and returns the corresponding 3D model from our reference dataset.

## Configuration

Adjust the parameters in the respective notebooks (`createFullDataset.ipynb`, `classification.ipynb`, `createLatentVector.ipynb`, and `retrieval.ipynb`) to customize the behavior of the data generation, model training, latent vector creation, and retrieval processes.

## About the Project


- **Development Period:**
  - Worked on this project in 2023 during my Master's in MMVAI at Paris Saclay.

## Issues and Contributions

If you encounter any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## Acknowledgments

This project relies on the McGill dataset, and we acknowledge the creators and contributors to the dataset for their valuable resource.

