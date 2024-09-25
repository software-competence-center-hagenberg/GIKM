import time
import numpy as np
import os
import scipy
import pandas as pd
from func import Classifier, predictionClassifier, combineMultipleClassifiers


def run_experiments_mnist():
    # Construct the file path for the .mat file
    # Assuming the current working directory is the Python equivalent of MATLAB's pwd
    print(os.getcwd())
    dataset_path = os.path.join(os.getcwd(), 'Datasets', 'MNIST', 'mnist_all.mat')
    
    # Load the .mat file
    data = scipy.io.loadmat(dataset_path)
    
    # Extract and convert train and test data to double (float) and normalize
    train0 = data['train0']
    train1 = data['train1']
    train2 = data['train2']
    train3 = data['train3']
    train4 = data['train4']
    train5 = data['train5']
    train6 = data['train6']
    train7 = data['train7']
    train8 = data['train8']
    train9 = data['train9']

    test0 = data['test0']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']
    test4 = data['test4']
    test5 = data['test5']
    test6 = data['test6']
    test7 = data['test7']
    test8 = data['test8']
    test9 = data['test9']
    
    # Concatenate all training data and normalize by dividing by 255
    y_data_trn = np.hstack([
        train0.T, train1.T, train2.T, train3.T, train4.T,
        train5.T, train6.T, train7.T, train8.T, train9.T
    ]).astype(np.float64) / 255.0
    
    # Print the shape of the training data for verification
    print(f"shape of train0: {train0.shape}")
    print("Shape of y_data_trn:", y_data_trn.shape)

    # Similarly, load test data if needed
    # test0 = data['test0']
    # test1 = data['test1']
    # ...
    # and concatenate them as needed
    y_data_trn = np.tanh(y_data_trn)

    # Create labels for training data
    labels_trn = np.hstack([
        1 * np.ones(train0.shape[0]),
        2 * np.ones(train1.shape[0]),
        3 * np.ones(train2.shape[0]),
        4 * np.ones(train3.shape[0]),
        5 * np.ones(train4.shape[0]),
        6 * np.ones(train5.shape[0]),
        7 * np.ones(train6.shape[0]),
        8 * np.ones(train7.shape[0]),
        9 * np.ones(train8.shape[0]),
        10 * np.ones(train9.shape[0])
    ]).astype(int)

    # Convert test data to double (float64) and normalize
    y_data_test = np.hstack([
        test0.T, test1.T, test2.T, test3.T, test4.T,
        test5.T, test6.T, test7.T, test8.T, test9.T
    ]).astype(np.float64) / 255.0

    # Apply tanh activation function
    y_data_test = np.tanh(y_data_test)

    # Create labels for test data
    labels_test = np.hstack([
        1 * np.ones(test0.shape[0]),
        2 * np.ones(test1.shape[0]),
        3 * np.ones(test2.shape[0]),
        4 * np.ones(test3.shape[0]),
        5 * np.ones(test4.shape[0]),
        6 * np.ones(test5.shape[0]),
        7 * np.ones(test6.shape[0]),
        8 * np.ones(test7.shape[0]),
        9 * np.ones(test8.shape[0]),
        10 * np.ones(test9.shape[0])
    ]).astype(int)

    # Unique labels and number of labels
    labels = np.unique(labels_trn)
    Q = len(labels)

    # Initialize lists to store results for each class
    min_distance_arr = [None] * Q
    labels_arr_arr = [None] * Q
    max_modeling_error = 0
    for i in range(Q):
        # Find indices for the current label
        ind = np.where(labels_trn == labels[i])[0]

        # Train the classifier for the current class data
        CLF = Classifier(y_data_trn[:, ind], labels_trn[ind], subspace_dim=20, Nb=1000)

        # Perform prediction on the test data
        min_distance_arr[i], labels_arr_arr[i] = predictionClassifier(y_data_test, CLF)

        # Track the maximum modeling error encountered
        max_modeling_error = max(max_modeling_error, CLF['max_modeling_error'])

    # Combine results from multiple classifiers
    _, hat_labels_test = combineMultipleClassifiers(min_distance_arr, labels_arr_arr)

    # Calculate accuracy of the predictions
    acc = np.mean(hat_labels_test == labels_test)

    print(f"Overall maximum modeling error: {max_modeling_error}")
    print(f"Accuracy: {acc}")
    data = {
    'Metric': ['Max Modelling Error', 'Accuracy'],
    'Value': [max_modeling_error, acc]
}
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = "./results/accuracy_metrics_mnist_compare.csv"
    df.to_csv(csv_filename, index=False)



def main():
    start_time = time.time()  # Record the start time
    run_experiments_mnist()    # Run the experiments
    end_time = time.time()      # Record the end time

    execution_time = end_time - start_time  # Calculate execution time
    print(f"Execution time: {execution_time:.2f} seconds")  # Print execution time

if __name__ == "__main__":
    main()