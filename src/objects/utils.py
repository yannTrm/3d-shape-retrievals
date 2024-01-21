# -*- coding: utf-8 -*-
# Import 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import os
import json
import numpy as np

import torch

from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

def process_dataset(dataset, num_steps_x=6, num_steps_y=6):
    """
    Process a dataset, generate 2D views, and save them to the corresponding directories.

    Parameters:
    - dataset: The dataset to process.
    - num_steps_x: The number of steps along the x-axis for generating 2D views.
    - num_steps_y: The number of steps along the y-axis for generating 2D views.
    """

    for i in tqdm(range(len(dataset)), desc="Processing dataset"):
        sample = dataset[i]

        # Extract the relevant part of the path
        path = sample['path']
        category = sample['label']
        base_path = os.path.join(dataset.root_dir, category)

        # Create the Image2D subdirectory if it doesn't exist
        file_name_without_extension = os.path.splitext(os.path.basename(path))[0]
        file_name_without_extension = os.path.splitext(os.path.basename(file_name_without_extension))[0]
        image2d_subdir = os.path.join(base_path, f'{category}Image2D', file_name_without_extension)
        os.makedirs(image2d_subdir, exist_ok=True)

        views, sample_data = dataset.generate_2d_views(i, num_steps_x, num_steps_y)

        # Save the images in the image2d_subdir
        for idx, image in enumerate(views):
            image_filename = f"{file_name_without_extension}_view_{idx + 1}.png"
            image_path = os.path.join(image2d_subdir, image_filename)
            image.save(image_path)
        print(f"{image_path} successfully saved ")

def split_dataset(dataset, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - dataset (Dataset): The dataset to be split.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Seed for random state for reproducibility.

    Returns:
    - train_dataset (Dataset): Training set.
    - test_dataset (Dataset): Testing set.
    """
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=[dataset.get_category_from_path(dataset.all_data_paths[i]) for i in range(len(dataset))],
        random_state=random_state
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, test_dataset


def split_dataset_1(dataset, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - dataset (Dataset): The dataset to be split.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Seed for random state for reproducibility.

    Returns:
    - train_dataset (Dataset): Training set.
    - test_dataset (Dataset): Testing set.
    """
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=test_size,
        stratify=dataset.labels,  # Ensures proportional class distribution in train and test sets
        random_state=random_state
    )

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    return train_dataset, test_dataset

def generate_latent_vectors(train_dataset, features_extractor, transform, device):
    """
    Generate latent vectors for 3D models from a given train dataset.

    Args:
    - train_dataset (Dataset): The dataset containing 3D models.
    - features_extractor (torch.nn.Module): The model to extract latent vectors.
    - transform (torchvision.transforms.Transform): Preprocessing for input images.
    - device (torch.device): Device to perform computations (e.g., 'cuda' or 'cpu').

    Returns:
    None
    """
    # Iterate through the train_dataset
    for idx in tqdm(range(len(train_dataset))):
        # Access the sample in the dataset
        sample = train_dataset[idx]
        label = sample['label']
        path_2D = sample['path_2d_data_folder']

        # Split the path into the head (all but the last component) and tail (the last component)
        head, model3d = os.path.split(path_2D)  # model3d corresponds to the 3D model name

        to_save_directory, tail = os.path.split(head) 
        image_files = [f for f in os.listdir(path_2D) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        # Create or ensure the existence of the 'latent' directory
        latent_directory = os.path.join(to_save_directory, 'latent')
        if not os.path.exists(latent_directory):
            os.makedirs(latent_directory)

        # Dictionary to store latent vectors for each 3D model
        latent_vectors_dict = {}

        # Loop over all images related to one 3D model
        for image_file in image_files:
            image_path = os.path.join(path_2D, image_file)

            input_image = Image.open(image_path).convert('RGB')
            input_tensor = transform(input_image).unsqueeze(0).to(device)

            # Forward pass through the latent vector extractor
            with torch.no_grad():
                latent_vector = features_extractor(input_tensor).cpu().numpy()
            latent_vectors_dict[image_file] = latent_vector.flatten().tolist()

        # Save or append the latent vectors dictionary to the JSON file in write mode ('w')
        output_json_path = os.path.join(latent_directory, f"{model3d}LatentVector.json")
        with open(output_json_path, 'w') as json_file:
            json.dump(latent_vectors_dict, json_file)

def euclidean_distance(vector1, vector2):
    """
    Calculate the Euclidean distance between two vectors.

    Parameters:
    - vector1: First vector (list, numpy array, or similar).
    - vector2: Second vector (list, numpy array, or similar).

    Returns:
    - Euclidean distance between the two vectors.
    """
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Ensure vectors are of the same length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length for Euclidean distance calculation.")

    # Calculate Euclidean distance
    distance = np.linalg.norm(vector1 - vector2)

    return distance


def read_json_data(data_directory):
    """
    Reads JSON files from a directory structure and stores the data in a dictionary.

    Parameters:
    - data_directory (str): The path to the main data directory.

    Returns:
    dict: A dictionary where keys are the file paths and values are the contents of the JSON files.
    """
    # Initialize the dictionary to store the data
    data_dict = {}

    # Iterate through subfolders (classes)
    classes = os.listdir(data_directory)
    for class_folder in classes:
        class_path = os.path.join(data_directory, class_folder)

        # Iterate through latent folders in each class
        for latent_folder in os.listdir(class_path):
            latent_path = os.path.join(class_path, latent_folder)

            # Iterate through JSON files in the latent folder
            for json_file in os.listdir(latent_path):
                json_path = os.path.join(latent_path, json_file)

                # Check if the file is a JSON file
                if json_file.endswith(".json"):
                    # Read the content of the JSON file and store it in the dictionary
                    with open(json_path, 'r') as file:
                        json_content = json.load(file)
                        data_dict[json_path] = json_content
    
    return data_dict


def find_closest_image_info(
    dataset_latent_vectors: Dict[str, Dict[str, List[float]]],
    latent_vector: List[float],
    first_image_path: str
) -> Tuple[float, str, str, str]:
    """
    Finds the closest image information in the dataset_latent_vectors for a given latent vector.

    Parameters:
    - dataset_latent_vectors (Dict[str, Dict[str, List[float]]]): Latent vectors for images in the dataset.
    - latent_vector (List[float]): Latent vector for the query image.
    - first_image_path (str): Path of the query image.

    Returns:
    - Tuple[float, str, str, str]: Information for the closest image.
      Contains (distance, json_file, image, first_image_path).
      Image is the one in our database
    """
    min_distance = float('inf') 
    result_info = []

    for json_file, model_3d in dataset_latent_vectors.items():
        for image, liste in model_3d.items():
            distance = euclidean_distance(liste, latent_vector)
            
            # Check if the current distance is smaller than the minimum
            if distance < min_distance:
                min_distance = distance
                result_info = [distance, json_file, image, first_image_path]

    return tuple(result_info)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------