import os
import numpy as np
import scipy.io
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.preprocessing import LabelEncoder
from func import divide_data_into_non_iid_label_screw, Classifier, predictionClassifier
import pandas as pd


def run_experiments_cifar10():
    # Set the random seed
    np.random.seed(4232)

    # Define the root folder
    dataset_folder = os.path.join(os.getcwd(),'Datasets', 'cifar10')

    # Load training and testing images (assuming folders 'train' and 'test')
    training_dataset = image_dataset_from_directory(os.path.join(dataset_folder, 'train'),
                                                    label_mode='int', image_size=(224, 224), shuffle=False)
    testing_dataset = image_dataset_from_directory(os.path.join(dataset_folder, 'test'),
                                                   label_mode='int', image_size=(224, 224), shuffle=False)

    # Initialize the feature matrix for training data
    num_training_files = len(training_dataset.file_paths)
    y_data_trn = np.zeros((2048, num_training_files))
    prefix = "._"
    # Load features for training data
    for i, file_path in enumerate(training_dataset.file_paths):
        base_name, _ = os.path.splitext(os.path.basename(file_path))
        if not base_name.startswith(prefix):
            mat_file = os.path.join(os.path.dirname(file_path), f'{base_name}_resnet50.mat')
            #print(f'Reading feature of file = {file_path}')
            resnet50_features = scipy.io.loadmat(mat_file)['resnet50_features']
            y_data_trn[:, i] = resnet50_features.flatten()

    # Apply tanh activation
    y_data_trn = np.tanh(y_data_trn)

    # Get training labels
    training_labels = np.concatenate([labels for _, labels in training_dataset], axis=0)
    classes = np.unique(training_labels)
    C = len(classes)

    # Initialize the feature matrix for testing data
    num_testing_files = len(testing_dataset.file_paths)
    y_data_test = np.zeros((2048, num_testing_files))
    labels_trn = training_labels
    # Load features for testing data
    for i, file_path in enumerate(testing_dataset.file_paths):
        base_name, _ = os.path.splitext(os.path.basename(file_path))
        mat_file = os.path.join(os.path.dirname(file_path), f'{base_name}_resnet50.mat')
        #print(f'Reading feature of file = {file_path}')
        resnet50_features = scipy.io.loadmat(mat_file)['resnet50_features']
        y_data_test[:, i] = resnet50_features.flatten()

    # Apply tanh activation
    y_data_test = np.tanh(y_data_test)

    # Get testing labels
    testing_labels = np.concatenate([labels for _, labels in testing_dataset], axis=0)

    # Map testing labels to indices
    labels_test = testing_labels

    n_clients = 100
    n_experiments = 3
    avg_local_acc_arr = np.zeros(n_experiments)
    avg_global_acc_arr = np.zeros(n_experiments)

    for k in range(n_experiments):
        client_id_trn = divide_data_into_non_iid_label_screw(labels_trn, n_clients, round(0.2 * C))
        local_acc_arr = np.zeros(n_clients)
        distance_matrix = np.full((C, y_data_test.shape[1]), np.inf)

        for j in range(n_clients):
            clf = Classifier(y_data_trn[:, client_id_trn == j+1], labels_trn[client_id_trn == j+1], 20, 1000)
            classes_client = np.unique(labels_trn[client_id_trn == j+1])

            # Select the test data belonging to the classes present in this client
            test_data_ind = np.hstack([np.where(labels_test == cls)[0] for cls in classes_client])
            y_data_test_client = y_data_test[:, test_data_ind]
            labels_test_client = labels_test[test_data_ind]

            distance_arr, labels_predicted = predictionClassifier(y_data_test_client, clf)
            local_acc_arr[j] = np.mean(labels_predicted == labels_test_client)

            for i, test_ind in enumerate(test_data_ind):
                distance_matrix[labels_predicted[i], test_ind] = min(distance_arr[i], distance_matrix[labels_predicted[i], test_ind])

        # Determine global predictions
        min_distance = np.min(distance_matrix, axis=0)
        hat_labels_test = np.zeros(y_data_test.shape[1], dtype=int)
        for i in range(C):
            hat_labels_test[distance_matrix[i, :] == min_distance] = i# + 1

        # Global accuracy calculation
        global_acc_arr = np.zeros(n_clients)
        for j in range(n_clients):
            classes_client = np.unique(labels_trn[client_id_trn == j+1])
            test_data_ind = np.hstack([np.where(labels_test == cls)[0] for cls in classes_client])
            global_acc_arr[j] = np.mean(hat_labels_test[test_data_ind] == labels_test[test_data_ind])

        avg_local_acc_arr[k] = np.mean(local_acc_arr)
        avg_global_acc_arr[k] = np.mean(global_acc_arr)

    # Compute final results
    mean_local_acc_20 = np.mean(avg_local_acc_arr)
    std_local_acc_20 = np.std(avg_local_acc_arr)
    mean_global_acc_20 = np.mean(avg_global_acc_arr)
    std_global_acc_20 = np.std(avg_global_acc_arr)

    print(f"Local accuracy (20%): {mean_local_acc_20:.6f}, std = {std_local_acc_20:.6f}")
    print(f"Global accuracy (20%): {mean_global_acc_20:.6f}, std = {std_global_acc_20:.6f}")
    data = {
    'Metric': ['Mean Local Accuracy (20%)', 'STD Local Accuracy (20%)', 
               'Mean Global Accuracy (20%)', 'STD Global Accuracy (20%)'],
    'Value': [mean_local_acc_20, std_local_acc_20, mean_global_acc_20, std_global_acc_20]
}
    df = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    csv_filename = "./results/accuracy_metrics_cifar10.csv"
    df.to_csv(csv_filename, index=False)


def main():
    run_experiments_cifar10()


if __name__ == "__main__":
    main()