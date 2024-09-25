import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.linalg import sqrtm, eig
from joblib import Parallel, delayed  # For parallelization



def divide_data_into_non_iid_label_screw(labels_trn, n_clients, n_classes_per_client):
    labels = np.unique(labels_trn)
    C = len(labels)

    # Random assignment of labels to clients
    tM = np.random.randint(low=min(labels), high=max(labels) + 1, size=(n_classes_per_client, n_clients))

    # Calculate groups per class
    n_groups_per_class_arr = np.array([np.sum(np.sum(tM == label, axis=0)) for label in labels])

    ind_cell = [np.where(labels_trn == label)[0] for label in labels]

    # Divide the data into groups for each class
    ind_cell_grouped = [
        np.split(np.random.permutation(ind), n_groups_per_class_arr[i]) if n_groups_per_class_arr[i] > 0 else []
        for i, ind in enumerate(ind_cell)
    ]

    # Assign indices to clients
    client_trn_ind = [[] for _ in range(n_clients)]

    for i, label in enumerate(labels):
        class_clients = np.where(tM == label)
        for j, (client_idx, _) in enumerate(zip(*class_clients)):
            client_trn_ind[client_idx].extend(ind_cell_grouped[i][j])

    # Create the client ID assignment array
    client_id_trn = np.zeros(len(labels_trn), dtype=int)

    for client_idx, indices in enumerate(client_trn_ind):
        client_id_trn[indices] = client_idx + 1

    return client_id_trn

def divide_data_into_groups(N, nGroup, random_flag=True):
    # Calculate base group sizes and the remainder
    nElementPerGroup = N // nGroup
    remainder = N % nGroup

    # Create group assignments
    group = np.repeat(np.arange(1, nGroup + 1), nElementPerGroup)

    # Add extra elements for the remainder
    if remainder > 0:
        extra_elements = np.arange(1, remainder + 1)
        group = np.concatenate([group, extra_elements])

    # Shuffle if random_flag is set
    if random_flag:
        np.random.shuffle(group)

    return group

def k_means_clustering(data, no_of_clusters):
    kmeans = KMeans(n_clusters=no_of_clusters, max_iter=10000, n_init=10, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

def dimReduce(y_data, n):
    N = y_data.shape[1]
    data = y_data - np.mean(y_data, axis=1, keepdims=True)
    covariance = (1/(N-1)) * np.dot(data, data.T)
    eig_val, PC = eig(covariance) 
    eig_val = eig_val.real
    PC = PC.real
    neig_val = -eig_val
    sorted_indices = np.argsort(neig_val)
    neg_eig_val = neig_val[sorted_indices]
    ind = np.where(neg_eig_val < -np.finfo(float).eps)[0]
    n = min(n, ind[-1] if ind.size else 0)
    PC = PC[:, sorted_indices[:n]]
    y_data_n_subspace = np.dot(PC.T, y_data)
    return y_data_n_subspace, PC


def KxxMatrix(x_matrix, weights_matrix, kerneltype):
    if kerneltype.lower() == 'gaussian':
        n, N = x_matrix.shape
        
        # Compute Wx, where Wx = W * x_matrix
        Wx = np.dot(weights_matrix, x_matrix)
        
        # Compute pairwise weighted squared distances
        dist_matrix = np.sum((Wx[:, :, None] - Wx[:, None, :]) ** 2, axis=0)
        
        # Apply the Gaussian kernel
        Kxx = np.exp(-0.5 * dist_matrix / n)
        
    return Kxx



def KxaMatrix(x_matrix, a_matrix, sqrt_weights_matrix, kerneltype):
    if kerneltype.lower() == 'gaussian':
        n = x_matrix.shape[0]

        # Weighted matrices
        Wx = np.dot(sqrt_weights_matrix, x_matrix)
        Wa = np.dot(sqrt_weights_matrix, a_matrix)

        # Compute pairwise Euclidean distances and apply the Gaussian kernel
        distMat = pairwise_distances(Wx.T, Wa.T, metric='euclidean')
        Kxa = np.exp(-(0.5 / n) * distMat**2)

    return Kxa



def kernel_regularized_least_squares(KernelMatrix, LabelsMatrix):
    N = LabelsMatrix.shape[1]
    tv0 = LabelsMatrix.size
    tv1 = np.sum(np.sum(LabelsMatrix**2, axis=0)) / tv0
    tau = 2 * tv1
    e = 0.5 * tv1
    lambda_ = tau + e
    lambda_ini = lambda_
    lambda_chg = 1
    itr_count = 0

    while (lambda_chg > 0.01) and (itr_count < 100):
        B = np.linalg.inv(lambda_ * np.eye(N) + KernelMatrix)
        temp_matrix = np.dot(KernelMatrix, B)
        tv = np.sum(np.sum((LabelsMatrix - np.dot(LabelsMatrix, temp_matrix.T))**2, axis=0))
        lambda_ = tau + (tv / tv0)
        lambda_chg = abs(lambda_ - lambda_ini)
        lambda_ini = lambda_
        itr_count += 1
    
    print(f'iterations = {itr_count}, regularization = {lambda_}, N = {N}.')
    return B

def AutoencoderFiltering(y_data_new, AE):
    # Project y_data_new onto AE's principal components
    x_data = np.dot(AE['PC'].T, y_data_new)
    N = x_data.shape[1]

    # Initialize hat_y_data based on the size of AE['y_data']
    hat_y_data = np.zeros((AE['y_data'].shape[0], N))

    # Define chunk size for processing large datasets
    chunk_size = 1000
    N_loops = int(np.ceil(N / chunk_size)) if N > chunk_size else 1

    # Loop over chunks to process data in smaller pieces (for large datasets)
    for loop in range(N_loops):
        ini_ind = chunk_size * loop
        fin_ind = min(N, chunk_size * (loop + 1))

        # Compute kernel matrix between new data and AE subspace
        Kxa = KxaMatrix(x_data[:, ini_ind:fin_ind], AE['y_data_n_subspace'], AE['sqrt_weights_matrix'], AE['kerneltype'])

        # Compute weights for reconstruction
        weights = np.dot(Kxa, AE['B'])
        sum_weights = np.sum(weights, axis=1, keepdims=True)

        # Normalize weights
        weights /= sum_weights

        # Reconstruct the data
        hat_y_data[:, ini_ind:fin_ind] = np.dot(AE['y_data'], weights.T)

    # Calculate the Euclidean distance between original and reconstructed data
    distance = np.sqrt(np.sum((y_data_new - hat_y_data)**2, axis=0))

    return hat_y_data, distance



def Autoencoder(y_data, subspace_dim):
    AE = {}

    # Use the optimized dimensionality reduction function
    AE['y_data_n_subspace'], AE['PC'] = dimReduce(y_data, subspace_dim)

    # Ensure subspace has a significant spread
    spread = np.max(AE['y_data_n_subspace'], axis=1) - np.min(AE['y_data_n_subspace'], axis=1)
    while np.min(spread) < 1e-3:
        subspace_dim -= 1
        AE['y_data_n_subspace'], AE['PC'] = dimReduce(y_data, subspace_dim)
        spread = np.max(AE['y_data_n_subspace'], axis=1) - np.min(AE['y_data_n_subspace'], axis=1)

    # Compute the weights matrix (inverse of covariance matrix)
    weights_matrix = np.linalg.inv(np.cov(AE['y_data_n_subspace']))

    # Define kernel type and compute the kernel matrix
    AE['kerneltype'] = 'Gaussian'
    Kxx = KxxMatrix(AE['y_data_n_subspace'], weights_matrix, AE['kerneltype'])

    # Perform kernel regularized least squares
    AE['B'] = kernel_regularized_least_squares(Kxx, y_data)

    # Store additional data for the autoencoder model
    AE['y_data'] = y_data
    AE['sqrt_weights_matrix'] = sqrtm(weights_matrix)

    # Call AutoencoderFiltering and compute the modeling error
    _, distance = AutoencoderFiltering(y_data, AE)
    AE['modeling_error'] = 1 - np.exp(-(1 / y_data.shape[0]) * np.max(distance))

    return AE

def parallelAutoencoders(y_data, subspace_dim, Nb):
    """
    Train multiple autoencoders on clustered subsets of the input data.

    Parameters:
    - y_data (numpy.ndarray): The input data.
    - subspace_dim (int): The subspace dimension for the autoencoder.
    - Nb (int): The number of clusters.

    Returns:
    - AE_arr (list): A list of trained autoencoders.
    """
    AE_arr = []
    N = y_data.shape[1]  # Number of samples
    S = round(N / Nb)  # Approximate number of clusters

    if S > 1:
        # Perform k-means clustering with k-means++ initialization
        cluster_labels = k_means_clustering(y_data.T, S)

        # Remove clusters with less than 30 samples
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        valid_clusters = unique_labels[counts >= 30]

        # Re-run k-means clustering with reduced number of clusters if necessary
        while len(valid_clusters) < S:
            S -= 1
            cluster_labels = k_means_clustering(y_data.T, S)
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            valid_clusters = unique_labels[counts >= 30]

        # Train autoencoders in parallel
        AE_arr = Parallel(n_jobs=-1)(delayed(Autoencoder)(
             y_data[:, cluster_labels == label], subspace_dim) for label in valid_clusters)

    else:
        # Train a single autoencoder if only one cluster is needed
        AE = Autoencoder(y_data, subspace_dim)
        AE_arr.append(AE)

    return AE_arr


def Classifier(y_data, label_data, subspace_dim, Nb):
    """
    Build a classifier using autoencoders for each class of the input data.

    Parameters:
    - y_data (numpy.ndarray): The input data (features).
    - label_data (numpy.ndarray): The labels corresponding to the input data.
    - subspace_dim (int): The subspace dimension for the autoencoders.
    - Nb (int): Parameter determining the number of clusters/subsets.

    Returns:
    - CLF (dict): A dictionary representing the classifier containing:
      - 'labels': Unique labels of the input data.
      - 'AE_arr_arr': List of lists, where each sublist contains autoencoders trained on a specific class.
      - 'max_modeling_error': Maximum modeling error encountered across all autoencoders.
    """
    CLF = {} # maybe update this to a class
    CLF['labels'] = np.unique(label_data)
    C = len(CLF['labels'])
    CLF['AE_arr_arr'] = [None] * C
    max_modeling_error = 0

    # Function to handle autoencoder training for each class
    def train_class_autoencoder(class_data, label, subspace_dim, Nb):
        print(f'Building an autoencoder for class = {label}...')
        AE_arr = parallelAutoencoders(class_data, subspace_dim, Nb)
        max_error = max(ae['modeling_error'] for ae in AE_arr)
        print(f'Maximum modeling error for class {label} = {max_error}.')
        return AE_arr, max_error

    # Extract data for each class and train autoencoders in parallel
    results = Parallel(n_jobs=-1)(delayed(train_class_autoencoder)(
        y_data[:, label_data == label], label, subspace_dim, Nb) for label in CLF['labels'])
    for i, (AE_arr, max_error) in enumerate(results):
        CLF['AE_arr_arr'][i] = AE_arr
        max_modeling_error = max(max_modeling_error, max_error)

    CLF['max_modeling_error'] = max_modeling_error
    print(f'Overall maximum modeling error = {max_modeling_error}.')

    return CLF


def combineMultipleAutoencoders(y_data_new, AE_arr):
    N = y_data_new.shape[1]  # Number of data points
    Q = len(AE_arr)  # Number of autoencoders
    distance_matrix = np.zeros((Q, N))  # Initialize a distance matrix

    # Compute the distance for each autoencoder
    for i in range(Q):
        _, distances = AutoencoderFiltering(y_data_new, AE_arr[i])
        distance_matrix[i, :] = distances

    distance_min = np.min(distance_matrix, axis=0)

    # Create a mask where distance_matrix matches distance_min
    mask = (distance_matrix == distance_min)

    # Use the mask to select the corresponding values from distance_matrix
    distance = np.where(mask, distance_matrix, np.inf).min(axis=0)

    return distance


def predictionClassifier(y_data_new, CLF):
    C = len(CLF['AE_arr_arr'])  # Number of unique classes
    N = y_data_new.shape[1]  # Number of data points
    distance_matrix = np.zeros((C, N))  # Initialize a distance matrix

    # Compute distances for all classes
    for i in range(C):
        distance_matrix[i, :] = combineMultipleAutoencoders(y_data_new, CLF['AE_arr_arr'][i])

    # Find the minimum distance and corresponding labels
    min_distance = np.min(distance_matrix, axis=0)
    labels_arr = CLF['labels'][np.argmin(distance_matrix, axis=0)]

    return min_distance, labels_arr


def combineMultipleClassifiers(distance_arr, labels_arr_arr):
    Q = len(distance_arr)  # Number of classifiers
    N = distance_arr[0].shape[0]  # Number of data points

    min_distance = np.full(N, np.inf)  # Initialize with infinity
    labels_arr = np.zeros(N, dtype=int)  # Initialize label array

    for i in range(Q):
        # Update min_distance with the element-wise minimum
        mask = distance_arr[i] < min_distance
        min_distance[mask] = distance_arr[i][mask]
        labels_arr[mask] = labels_arr_arr[i][mask]

    return min_distance, labels_arr