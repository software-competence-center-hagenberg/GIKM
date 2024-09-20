import os
import glob
import random
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

def get_data(folder_name, n_per_class, feature_type):
    # Define the path to the dataset folder
    base = os.getcwd()
    dataset_folder = os.path.join(os.getcwd(),'Datasets', 'office+caltech256', folder_name)

    # Get the subfolders (class names) and corresponding file paths
    classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
    classes.sort()  # Ensuring class labels are consistent

    # Create image list and labels
    images = []
    labels = []
    for i, class_name in enumerate(classes):
        class_folder = os.path.join(dataset_folder, class_name)
        class_images = glob.glob(os.path.join(class_folder, '*.jpg'))  # Assuming images are in .jpg format
        random.shuffle(class_images)
        images.extend(class_images[:n_per_class])  # Take n_per_class from each class
        labels.extend([i + 1] * min(n_per_class, len(class_images)))  # Assign class label

    # Split into training and testing sets
    train_files, test_files, train_labels, test_labels = train_test_split(images, labels, stratify=labels, test_size=0.3333)

    # Helper function to load features
    def load_features(image_file, feature_type):
        base, ext = os.path.splitext(image_file)
        mat_file = base + f'_{feature_type}.mat'
        feature_var = f'{feature_type}_features'
        mat_data = scipy.io.loadmat(mat_file)
        return mat_data[feature_var]

    # Load training data
    y_data_trn = np.asarray([load_features(file, feature_type) for file in train_files])
    y_data_trn = np.squeeze(y_data_trn)  # Convert list of arrays into a matrix

    # Load testing data
    y_data_test = np.asarray([load_features(file, feature_type) for file in test_files])
    y_data_test = np.squeeze(y_data_test)  # Convert list of arrays into a matrix

    return y_data_trn.T, np.array(train_labels), y_data_test.T, np.array(test_labels)