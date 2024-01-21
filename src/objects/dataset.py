# -*- coding: utf-8 -*-
# Import 
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import os
import io
import gzip
import random
import numpy as np
import matplotlib.pyplot as plt

import trimesh

from PIL import Image

from torch.utils.data import Dataset

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class McGillDataset(Dataset):
    """
    Custom dataset for the McGill dataset.

    Parameters:
    - root_dir (str): Root directory of the dataset.
    - file_type (str, optional): Type of file to load data from. Defaults to 'ply'.
    - seed (int, optional): Seed for shuffling. Defaults to None.

    Attributes:
    - s_PATH_3D_DATA_FILE (str): Constant representing the key for 3D data path in the sample dictionary.
    - s_PATH_2D_DATA_FOLDER (str): Constant representing the key for 2D data path in the sample dictionary.
    - s_LABEL (str): Constant representing the key for the label in the sample dictionary.

    Methods:
    - __init__(self, root_dir, file_type='ply', transform=None, seed=42): Constructor method.
    - __len__(self): Returns the total number of samples in the dataset.
    - __getitem__(self, idx): Returns a sample from the dataset.
    - get_3d_shape(self, idx): Extracts the 3D shape from the dataset.
    - get_category_from_path(self, data_path): Extracts category name from the data path.
    - get_all_data_paths(self): Returns a list of all data paths in the dataset.
    - get_path_2D(self, data_path): Gets the path to the folder containing 2D images.
    - get_2D_image(self, idx, idx_2d_img=0): Displays the first 2D image of the corresponding 3D model.
    """

    s_PATH_3D_DATA_FILE = "path_3d_data_file"
    s_PATH_2D_DATA_FOLDER = "path_2d_data_folder"
    s_LABEL = "label"
    
    def __init__(self, root_dir: str, file_type: str = 'ply', seed=42):
        """
        Custom dataset for McGill dataset.

        Parameters:
        - root_dir (str): Root directory of the dataset.
        - file_type (str, optional): Type of file to load data from. Defaults to 'ply'.
        - transform (callable, optional): Optional transform to be applied on a sample.
        - seed (int, optional): Seed for shuffling. Defaults to None.
        """
        self.root_dir = root_dir
        self.file_type = file_type

        # List all the categories (subdirectories) in the root directory
        self.categories = os.listdir(root_dir)
        self.class_to_idx = {'teddy': 0,
                            'ants': 1,
                            'crabs': 2,
                            'spectacles': 3,
                            'humans': 4,
                            'pliers': 5,
                            'spiders': 6,
                            'octopuses': 7,
                            'hands': 8,
                            'snakes': 9}

        # Create a dictionary to map category names to their respective subdirectories
        self.category_paths = {category: os.path.join(root_dir, category) for category in self.categories}

        self.all_data_paths = self.get_all_data_paths()



        random.seed(seed)
        random.shuffle(self.all_data_paths)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.all_data_paths)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - sample (dict): A dictionary containing 'data' (3D mesh),
          'label' (category name), 'path_3d_data_file' (path to 3D model),
          and 'path_2d_data_folder' (path to folder containing 2D images).
        """
        # Load the data (3D mesh) using the appropriate method
        data_path = self.all_data_paths[idx]
        category = self.get_category_from_path(data_path)


        # Get the path to the folder containing 2D images
        path_2D = self.get_path_2D(data_path)

        # Prepare the sample dictionary
        sample = {self.s_PATH_3D_DATA_FILE: data_path,  self.s_PATH_2D_DATA_FOLDER: path_2D, self.s_LABEL: self.class_to_idx[category]}

        return sample
    
    def get_all_data_paths(self):
        """
        Return a list of all data paths in the dataset.
        """
        all_paths = []
        for category in self.categories:
            category_path = self.category_paths[category]
            data_paths = os.listdir(os.path.join(category_path, f'{category}{self.file_type}'))
            full_paths = [os.path.join(category_path, f'{category}{self.file_type}', data_name) for data_name in data_paths]
            all_paths.extend(full_paths)
        return all_paths


    def get_category_from_path(self, data_path):
        """
        Extract category name from the data path.

        Parameters:
        - data_path (str): Full path to the data file.

        Returns:
        - category (str): Category name.
        """
        return os.path.basename(os.path.dirname(os.path.dirname(data_path)))

    def get_path_2D(self, data_path):
        """
        Get the path to the folder containing 2D images associated with the 3D model.

        Parameters:
        - data_path (str): Full path to the data file.

        Returns:
        - path_2D (str): Path to the folder containing 2D images.
        """
        category = self.get_category_from_path(data_path)
        file_name_without_extension = os.path.splitext(os.path.splitext(os.path.basename(data_path))[0])[0]
        path_2D = os.path.join(self.root_dir, category, f'{category}Image2D', file_name_without_extension)
        return path_2D


    def get_2D_image(self, idx, idx_2d_img= 0):
        """
        Display the first 2D image of the corresponding 3D model.

        Parameters:
        - idx (int): Index of the sample.
        """
        # Get the sample using the given index
        sample = self.__getitem__(idx)


        path_2D = sample[self.s_2D_DATA_PATH]

        image_files = os.listdir(path_2D)

        if image_files:
            # Sort the files to ensure consistent order
            image_files.sort()

                # Construct the path to the first image file
            first_image_path = os.path.join(path_2D, image_files[idx_2d_img])
                # Display the first 2D image
            img =  Image.open(first_image_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            return img
            
        else:
                print(f"No 2D images found for the 3D model at index {idx}")

    def count_samples_per_class(self):
        """
        Count the number of samples (3D models) for each class.

        Returns:
        - class_counts (dict): A dictionary where keys are class names and values are the corresponding sample counts.
        """
        class_counts = {}

        for sample in self:
            label = sample[self.s_LABEL]
            class_counts[label] = class_counts.get(label, 0) + 1

        return class_counts

    def get_3d_shape(self, idx):
        """
        Extract category name from the data path.

        Parameters:
        - data_path (str): Full path to the data file.

        Returns:
        - category (str): Category name.
        """
        sample = self.__getitem__(idx)
        data_path = sample['3d_data_path']
        if self.file_type == 'Im':
            with gzip.open(data_path, 'rb') as f:
                data = Image.open(io.BytesIO(f.read()))
        elif self.file_type == 'Ply':
            if data_path.endswith('.gz'):
                
                with gzip.open(data_path, 'rb') as f:
                    file_extension = os.path.splitext(data_path[:-3])[1]
                    data = trimesh.load(file_obj=io.BytesIO(f.read()), file_type=file_extension, process=False)
       
            else:
                file_extension = os.path.splitext(data_path)[1]
                data = trimesh.load(data_path, file_type=file_extension, process=False) 
        return data



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class McGillDataset3D(Dataset):
    def __init__(self, root_dir: str, file_type: str = 'ply', seed=42):
        """
        Custom dataset for McGill dataset.

        Parameters:
        - root_dir (str): Root directory of the dataset.
        - file_type (str, optional): Type of file to load data from. Defaults to 'ply'.
        - seed (int, optional): Seed for shuffling. Defaults to None.
        """
        self.root_dir = root_dir
        self.file_type = file_type

        # List all the categories (subdirectories) in the root directory
        self.categories = os.listdir(root_dir)

        # Create a dictionary to map category names to their respective subdirectories
        self.category_paths = {category: os.path.join(root_dir, category) for category in self.categories}

        self.all_data_paths = self.get_all_data_paths()

        random.seed(seed)
        random.shuffle(self.all_data_paths)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.all_data_paths)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.

        Parameters:
        - idx (int): Index of the sample.

        Returns:
        - sample (dict): A dictionary containing 'data' (3D mesh) and 'label' (category name).
        """
        # Load the data (3D mesh) using the appropriate method
        data_path = self.all_data_paths[idx]
        category = self.get_category_from_path(data_path)

        if self.file_type == 'Im':
            with gzip.open(data_path, 'rb') as f:
                data = Image.open(io.BytesIO(f.read()))
        elif self.file_type == 'Ply':
            if data_path.endswith('.gz'):
                with gzip.open(data_path, 'rb') as f:
                    file_extension = os.path.splitext(data_path[:-3])[1]
                    data = trimesh.load(file_obj=io.BytesIO(f.read()), file_type=file_extension, process=False)
            else:
                file_extension = os.path.splitext(data_path)[1]
                data = trimesh.load(data_path, file_type=file_extension, process=False)


        # Prepare the sample dictionary
        sample = {'data': data, 'label': category, 'path': data_path}

        return sample

    def get_category_from_path(self, data_path):
        """
        Extract category name from the data path.

        Parameters:
        - data_path (str): Full path to the data file.

        Returns:
        - category (str): Category name.
        """
        # Extract category name from the path
        category = os.path.basename(os.path.dirname(os.path.dirname(data_path)))
        return category


    def get_all_data_paths(self):
        """
        Return a list of all data paths in the dataset.
        """
        all_paths = []
        for category in self.categories:
            category_path = self.category_paths[category]
            data_paths = os.listdir(os.path.join(category_path, f'{category}{self.file_type}'))
            full_paths = [os.path.join(category_path, f'{category}{self.file_type}', data_name) for data_name in data_paths]
            all_paths.extend(full_paths)
        return all_paths

    def get_mesh(self, idx):
        """
        Return the 3D mesh.

        Parameters:
        - idx (int): Index of the sample.
        """
        return self.__getitem__(idx)['data']


    def generate_2d_views(self, idx, num_steps_x=5, num_steps_y=5, resolution=(256, 256)):
        """
        Generate 2D views from a 3D shape by rotating around both X and Y axes.

        Parameters:
        - idx (int): Index of the 3D shape in the dataset.
        - num_steps_x (int, optional): Number of steps in rotation around the X axis. Defaults to 5.
        - num_steps_y (int, optional): Number of steps in rotation around the Y axis. Defaults to 5.
        - resolution (tuple, optional): Resolution of the generated images. Defaults to (256, 256).

        Returns:
        - views (list): List of PIL Image objects representing 2D views.
        """
        # Load the data (3D mesh) using the existing __getitem__ method
        sample = self.__getitem__(idx)

        mesh = sample['data']

        # Create a scene with the mesh
        scene = mesh.scene()

        # Initialize a list to store generated views
        views = []

        # Calculate step sizes for rotation
        step_size_x = 360 / num_steps_x
        step_size_y = 180 / num_steps_y

        # Iterate through rotation angles
        for angle_x in range(0, 360, int(step_size_x)):
            for angle_y in range(-90, 90, int(step_size_y)):
                # Rotation matrix around the X-axis
                rotate_x = trimesh.transformations.rotation_matrix(
                    angle=np.radians(angle_x), direction=[1, 0, 0], point=scene.centroid)

                # Rotation matrix around the Y-axis
                rotate_y = trimesh.transformations.rotation_matrix(
                    angle=np.radians(angle_y), direction=[0, 1, 0], point=scene.centroid)

                # Combine the rotations
                rotate_combined = trimesh.transformations.concatenate_matrices(rotate_x, rotate_y)

                # Apply the combined transform to the camera view transform
                camera_old, _geometry = scene.graph[scene.camera.name]
                camera_new = np.dot(rotate_combined, camera_old)
                scene.graph[scene.camera.name] = camera_new

                # Render the scene and save the image
                try:
                    # Save a render of the object as a PNG

                    png = scene.save_image(resolution=resolution, visible=False)

                    # Convert the PNG to a PIL Image
                    image = Image.open(io.BytesIO(png))

                    # Append the image to the views list
                    views.append(image)
                except BaseException as e:
                    print(f"Unable to save image: {str(e)}")

        return views, sample

    def generate_2d_views_single_rotation(self, idx, num_steps=5, resolution=(256, 256)):
        """
        Generate 2D views from a 3D shape by rotating around the Z axis.

        Parameters:
        - idx (int): Index of the 3D shape in the dataset.
        - num_steps (int, optional): Number of steps in rotation around the Z axis. Defaults to 5.
        - resolution (tuple, optional): Resolution of the generated images. Defaults to (256, 256).

        Returns:
        - views (list): List of PIL Image objects representing 2D views.
        """
        # Load the data (3D mesh) using the existing __getitem__ method
        sample = self.__getitem__(idx)

        mesh = sample['data']

        # Create a scene with the mesh
        scene = mesh.scene()

        # Initialize a list to store generated views
        views = []

        # Calculate step size for rotation around the Z axis
        step_size_z = 360 / num_steps

        # Iterate through rotation angles around the Z axis
        for angle_z in range(0, 360, int(step_size_z)):
            # Rotation matrix around the Z-axis
            rotate_z = trimesh.transformations.rotation_matrix(
                angle=np.radians(angle_z), direction=[0, 1, 0], point=scene.centroid)

            # Apply the rotation transform to the camera view transform
            camera_old, _geometry = scene.graph[scene.camera.name]
            camera_new = np.dot(rotate_z, camera_old)
            scene.graph[scene.camera.name] = camera_new

            # Render the scene and save the image
            try:
                # Save a render of the object as a PNG
                png = scene.save_image(resolution=resolution, visible=False)

                # Convert the PNG to a PIL Image
                image = Image.open(io.BytesIO(png))

                # Append the image to the views list
                views.append(image)
            except BaseException as e:
                print(f"Unable to save image: {str(e)}")

        return views, sample





class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Custom dataset for loading 2D images from the specified folder structure.

        Parameters:
        - root_dir (str): The root directory containing subfolders for each class.
        - transform (callable, optional): Optional transform to be applied to the images.
        """
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform

        self.filepaths, self.labels = self.load_dataset()

    def load_dataset(self):
        filepaths = []
        labels = []
        for class_folder in self.classes:
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    filepath = os.path.join(class_path, filename)
                    filepaths.append(filepath)
                    labels.append(self.class_to_idx[class_folder])

        return filepaths, labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label






class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, shuffle=True):
        """
        Custom dataset for loading 2D images from the specified folder structure.

        Parameters:
        - root_dir (str): The root directory containing subfolders for each class.
        - transform (callable, optional): Optional transform to be applied to the images.
        - shuffle (bool): If True, shuffle the order of images in the dataset.
        """
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.transform = transform
        self.shuffle = shuffle

        self.filepaths, self.labels = self.load_dataset()

        if self.shuffle:
            self.shuffle_dataset()


    def load_dataset(self):
        filepaths = []
        labels = []
        for class_folder in self.classes:
            class_path = os.path.join(self.root_dir, class_folder)
            if os.path.isdir(class_path):
                for model_folder in os.listdir(class_path):
                    model_path = os.path.join(class_path, model_folder)
                    if os.path.isdir(model_path):
                        for filename in os.listdir(model_path):
                            filepath = os.path.join(model_path, filename)
                            filepaths.append(filepath)
                            labels.append(self.class_to_idx[class_folder])

        return filepaths, labels
    


    def shuffle_dataset(self):
        combined = list(zip(self.filepaths, self.labels))
        random.shuffle(combined)
        self.filepaths[:], self.labels[:] = zip(*combined)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label
    
    def plot_image(self, idx):
        """
        Plot the image at the specified index along with its label.

        Parameters:
        - idx (int): Index of the image in the dataset.
        """
        img, label = self.__getitem__(idx)

        # Convert tensor to numpy array
        img_np = img.permute(1, 2, 0).numpy()

        # Plot the image
        plt.imshow(img_np)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()