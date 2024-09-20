import os
import glob
import numpy as np
import scipy.io
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.preprocessing import LabelEncoder
from func import combineMultipleClassifiers, Classifier, predictionClassifier
import pandas as pd
from scipy.io import loadmat



def run_experiments_grocery():
    # Set the random seed
    np.random.seed(4232)
    images_folder = os.path.join(os.getcwd(),'Datasets', 'FreiburgGrocery')
        # Get the class names from the folder structure (i.e., subdirectories of 'FreiburgGrocery')
    classes = [d for d in os.listdir(images_folder) if os.path.isdir(os.path.join(images_folder, d))]
    
    # Read train and test datasets from the txt files
    train_df = pd.read_csv('GIKM_python/trainGrocery.txt', delimiter=' ', header=None)
    test_df = pd.read_csv('GIKM_python/testGrocery.txt', delimiter=' ', header=None)
    
    # Map training and testing file names to their full paths
    train_files = [os.path.join(images_folder, fname) for fname in train_df[0]]
    test_files = [os.path.join(images_folder, fname) for fname in test_df[0]]
    
    # Initialize arrays for training data
    y_data_trn = np.zeros((2048, len(train_files)))
    labels_trn = np.zeros((len(train_files),), dtype=int)

    for i, filepath in enumerate(train_files):
        print(f"Reading feature of file = {filepath}")
        
        # Load resnet50 features from the .mat file
        file = filepath.replace("/","\\")
        mat = loadmat(file.replace(".png",'_resnet50.mat'))
        resnet50_features = mat['resnet50_features']
        
        y_data_trn[:, i] = resnet50_features.flatten()
        
        # Get the class label based on the parent directory name
        class_name = os.path.basename(os.path.dirname(filepath))
        labels_trn[i] = classes.index(class_name)
    
    y_data_trn = np.tanh(y_data_trn)
    
    
    # Initialize arrays for testing data
    y_data_test = np.zeros((2048, len(test_files)))
    labels_test = np.zeros((len(test_files),), dtype=int)
    for i, filepath in enumerate(test_files):
        print(f"Reading feature of file = {filepath}")
        
        # Load resnet50 features from the .mat file
        file = filepath.replace("/","\\")
        mat = loadmat(file.replace(".png",'_resnet50.mat'))
        resnet50_features = mat['resnet50_features']
        
        y_data_test[:, i] = resnet50_features.flatten()
        
        # Get the class label based on the parent directory name
        class_name = os.path.basename(os.path.dirname(filepath))
        labels_test[i] = classes.index(class_name)
    
    # Apply tanh to testing data
    y_data_test = np.tanh(y_data_test)
    
    # Simulate number of clients as number of classes
    unique_labels = np.unique(labels_trn)
    Q = len(unique_labels)
    min_distance_arr = [None] * Q
    labels_arr_arr = [None] * Q
    max_modeling_error = 0
    
    for i in range(Q):
        ind = np.where(labels_trn == unique_labels[i])[0]
        
        clf = Classifier(y_data_trn[:, ind], labels_trn[ind], 20, 1000)
        min_distance_arr[i], labels_arr_arr[i] = predictionClassifier(y_data_test, clf)
        
        max_modeling_error = max(max_modeling_error, clf["max_modeling_error"])
    
    _, hat_labels_test = combineMultipleClassifiers(min_distance_arr, labels_arr_arr)
    
    # Calculate global accuracy
    acc = np.mean(hat_labels_test == labels_test)
    print(f"Global accuracy = {acc}, Maximum modeling error = {max_modeling_error}")
    
    data = {
    'Metric': ['Max Modelling Error', 'Accuracy'],
    'Value': [max_modeling_error, acc]
}
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = "./results/accuracy_metrics_grocery.csv"
    df.to_csv(csv_filename, index=False)

#     # Save results
#     #savemat('run_experiments_grocery.mat', {'acc': acc, 'max_modeling_error': max_modeling_error})


def main():
    run_experiments_grocery()


if __name__ == "__main__":
    main()