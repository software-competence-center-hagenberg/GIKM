import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import os
import scipy
from keras import datasets
from func import Classifier, predictionClassifier, combineMultipleClassifiers, divide_data_into_non_iid_label_screw
import pandas as pd




def run_experiments_fmnist():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train_flattened = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])

# Step 2: Transpose to get shape (784, 100000)
    y_data_trn = x_train_flattened.T /255
    #y_data_trn = np.zeros((x_train.shape[0] * x_train.shape[1], x_train.shape[0]))

    # Flatten and normalize the training images

    # Apply the tanh activation function
    y_data_trn = np.tanh(y_data_trn)

    # Process the training labels
    labels = np.unique(y_train)
    labels_trn = y_train.T



    # Initialize y_data_test
    x_test_flattened = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

# Step 2: Transpose to get shape (784, 100000)
    y_data_test = x_test_flattened.T /255

    # Apply the tanh activation function
    y_data_test = np.tanh(y_data_test)

    # Process the test labels
    labels_test = y_test.T

    # Additional variables for experiments
    n_clients = 100
    n_experiments = 3
    avg_local_acc_arr = np.zeros(n_experiments)
    avg_global_acc_arr = np.zeros(n_experiments)
    for k in range(n_experiments):
        n_samples_per_client = round(0.3 * len(labels))
        client_id_trn = divide_data_into_non_iid_label_screw(labels_trn, n_clients, n_samples_per_client)
        local_acc_arr = np.zeros(n_clients)
        distance_matrix = np.full((len(labels), y_data_test.shape[1]), np.inf)
        for j in range(n_clients):
            test = client_id_trn == (j+1)
            sub_data = y_data_trn[:,test]
            sub_labels = labels_trn[client_id_trn == (j+1)]
            CLF = Classifier(sub_data,sub_labels , 20, 1000)
            classes_client = np.unique(labels_trn[client_id_trn == (j+1)])
            test_data_ind = []
            for i in range(len(classes_client)):
                test_data_ind.extend(np.where(labels_test == classes_client[i])[0])
            
            y_data_test_client = y_data_test[:, test_data_ind]
            labels_test_client = labels_test[test_data_ind]
            distance_arr, labels_predicted = predictionClassifier(y_data_test_client, CLF)
            local_acc_arr[j] = np.mean(labels_predicted == labels_test_client)
            
            for i in range(len(test_data_ind)):
                distance_matrix[labels_predicted[i], test_data_ind[i]] = min(distance_arr[i], distance_matrix[labels_predicted[i], test_data_ind[i]])
        
        min_distance = np.min(distance_matrix, axis=0)
        hat_labels_test = np.zeros(y_data_test.shape[1], dtype=int)
        for i in range(len(labels)):
            hat_labels_test[distance_matrix[i, :] == min_distance] = i# + 1
        
        global_acc_arr = np.zeros(n_clients)
        for j in range(n_clients):
            classes_client = np.unique(labels_trn[client_id_trn == (j+1)])
            test_data_ind = []
            for i in range(len(classes_client)):
                test_data_ind.extend(np.where(labels_test == classes_client[i])[0])
            
            global_acc_arr[j] = np.mean(hat_labels_test[test_data_ind] == labels_test[test_data_ind])
        
        avg_local_acc_arr[k] = np.mean(local_acc_arr)
        avg_global_acc_arr[k] = np.mean(global_acc_arr)

    mean_local_acc_30 = np.mean(avg_local_acc_arr)
    std_local_acc_30 = np.std(avg_local_acc_arr)
    mean_global_acc_30 = np.mean(avg_global_acc_arr)
    std_global_acc_30 = np.std(avg_global_acc_arr)

    print(f'Mean Local Accuracy (30%): {mean_local_acc_30:.4f}')
    print(f'STD Local Accuracy (30%): {std_local_acc_30:.4f}')
    print(f'Mean Global Accuracy (30%): {mean_global_acc_30:.4f}')
    print(f'STD Global Accuracy (30%): {std_global_acc_30:.4f}')
    data = {
    'Metric': ['Mean Local Accuracy (30%)', 'STD Local Accuracy (30%)', 
               'Mean Global Accuracy (30%)', 'STD Global Accuracy (30%)'],
    'Value': [mean_local_acc_30, std_local_acc_30, mean_global_acc_30, std_global_acc_30]
}
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = "./results/accuracy_metrics_fmnist2.csv"
    df.to_csv(csv_filename, index=False)

    print(f'Saved to {csv_filename}')



def main():
    run_experiments_fmnist()


if __name__ == "__main__":
    main()