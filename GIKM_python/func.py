import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

def divide_data_into_non_iid_label_screw(labels_trn, n_clients, n_classes_per_client):
    labels = np.unique(labels_trn)
    C = len(labels)
    n_groups_per_class_arr = np.zeros(C, dtype=int)
    
    # Ensure that no class has zero groups assigned
    while np.sum(n_groups_per_class_arr == 0) > 0:
        tM = np.random.randint(min(labels), max(labels) + 1, (n_classes_per_client, n_clients))
        for i in range(C):
            n_groups_per_class_arr[i] = np.sum(np.sum(tM == labels[i], axis=0))
    
    ind_cell_cell = [[] for _ in range(C)]
    for i in range(C):
        ind = np.where(labels_trn == labels[i])[0]
        group = divide_data_into_groups(len(ind), n_groups_per_class_arr[i], True)
        ind_cell_cell[i] = [[] for _ in range(n_groups_per_class_arr[i])]
        for j in range(n_groups_per_class_arr[i]):
            ind_cell_cell[i][j] = ind[np.array(group) == (j + 1)]
    
    client_trn_ind_cell_cell = [[[] for _ in range(n_clients)] for _ in range(n_classes_per_client)]
    for i in range(C):
        a, b = np.where(tM == labels[i])
        for j in range(len(a)):
            client_trn_ind_cell_cell[a[j]][b[j]] = ind_cell_cell[i][j]
    
    client_id_trn = np.zeros(len(labels_trn), dtype=int)
    for i in range(n_clients):
        for j in range(n_classes_per_client):
            if len(client_trn_ind_cell_cell[j][i]) > 0:
                client_id_trn[client_trn_ind_cell_cell[j][i]] = i + 1

    return client_id_trn





def divide_data_into_groups(N, nGroup, random_flag):
    nElementPerGroup = N // nGroup
    group = np.repeat(np.arange(1, nGroup + 1), nElementPerGroup).tolist()

    # Add extra elements if N is not perfectly divisible by nGroup
    if len(group) < N:
        group.extend([nGroup] * (N - len(group)))
    
    # Shuffle the group array if random_flag is set to True
    if random_flag:
        np.random.shuffle(group)
    
    return group

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
    CLF = {}
    CLF['labels'] = np.unique(label_data)
    C = len(CLF['labels'])
    CLF['AE_arr_arr'] = [None] * C
    max_modeling_error = 0

    for i in range(C):
        label = CLF['labels'][i]
        print(f'Building an autoencoder for class = {label}...')
        
        # Extract data for the current class
        class_data = y_data[:, label_data == label]
        # Train autoencoders for the current class
        CLF['AE_arr_arr'][i] = parallelAutoencoders(class_data, subspace_dim, Nb)
        
        # Determine the maximum modeling error for the autoencoders
        for ae in CLF['AE_arr_arr'][i]:
            max_modeling_error = max(max_modeling_error, ae['modeling_error'])
        
        print(f'maximum modeling error for class {label} = {max_modeling_error}.')

    CLF['max_modeling_error'] = max_modeling_error
    print(f'overall maximum modeling error = {max_modeling_error}.')

    return CLF


def parallelAutoencoders(y_data, subspace_dim, Nb):
    """
    Train multiple autoencoders sequentially on clustered subsets of the input data.
    
    Parameters:
    - y_data (numpy.ndarray): The input data.
    - subspace_dim (int): The subspace dimension for the autoencoder.
    - Nb (int): A parameter to determine the number of clusters (subsets).
    
    Returns:
    - AE_arr (list): A list containing trained autoencoders.
    """
    AE_arr = []
    N = y_data.shape[1]  # Number of samples
    S = round(N / Nb)  # Determine number of clusters

    if S > 1:
        cluster_labels = k_means_clustering(y_data, S)
        labels_indices_less = []

        # Find clusters with fewer than 2 samples and filter out
        for i in range(S):
            ind = np.where(cluster_labels == i)[0]
            if len(ind) < 2:
                labels_indices_less.append(i)
        
        while labels_indices_less:
            S -= 1
            cluster_labels = k_means_clustering(y_data, S)
            labels_indices_less = []

            for i in range(S):
                ind = np.where(cluster_labels == i)[0]
                if len(ind) < 30:
                    labels_indices_less.append(i)
        
        # Sequential training of autoencoders
        for i in range(S):
            # Extract data for the current cluster
            cluster_data = y_data[:, cluster_labels == i]
            # Train autoencoder on the cluster data
            AE = Autoencoder(cluster_data, subspace_dim)
            # Append the trained autoencoder to the list
            AE_arr.append(AE)

    else:
        # Train a single autoencoder if only one cluster is needed
        AE = Autoencoder(y_data, subspace_dim)
        AE_arr.append(AE)
    
    return AE_arr


def k_means_clustering(data, no_of_clusters):
    kmeans = KMeans(n_clusters=no_of_clusters, max_iter=10000, n_init=10, random_state=0)
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

def dimReduce_fast(y_data, n):
    pca = PCA(n_components=n)
    y_data_n_subspace = pca.fit_transform(y_data.T).T
    PC = pca.components_
    
    return y_data_n_subspace, PC

def dimReduce_slow(y_data,n):
    y_data = y_data.T
    N = y_data.shape[1]
    data = y_data - np.mean(y_data, axis=1, keepdims=True)
    covariance = (1/(N-1)) * np.dot(data, data.T)
    #DF = pd.DataFrame(covariance)
    # save the dataframe as a csv file
    #DF.to_csv("covariance_py_dimreduce.csv")
    eig_val, PC = np.linalg.eig(covariance.T) # numpy wrong answers
    #eig_val, PC = scipy.linalg.eig(covariance) # recheck all of this code
    neig_val = -eig_val
    sorted_indices = np.argsort(neig_val)
    neg_eig_val = neig_val[sorted_indices]
    ind = np.where(neg_eig_val < -np.finfo(float).eps)[0]
    n = min(n, ind[-1] if ind.size else 0)
    PC = PC[:, sorted_indices[:n]]
    y_data_n_subspace = np.dot(PC.T, y_data)
    return y_data_n_subspace, PC
def KxxMatrix(x_matrix, weights_matrix, kerneltype):
    # x_matrix is n x N
    x_matrix = x_matrix.T
    # weights_matrix is n x 1

    if kerneltype.lower() == 'gaussian':
        n, N = x_matrix.shape
        Kxx = np.zeros((N, N))
        
        for i in range(N-1):
            for j in range(i+1, N):
                del_vec =  x_matrix[:, i] - x_matrix[:, j]
                Kxx[i, j] = np.dot(del_vec.T, np.dot(weights_matrix, del_vec))
                Kxx[j, i] = Kxx[i, j]
                
        Kxx = np.exp(-(0.5 / n) * Kxx)
        
    return Kxx

def KxaMatrix(x_matrix, a_matrix, sqrt_weights_matrix, kerneltype):
    if kerneltype.lower() == 'gaussian':
        n = x_matrix.shape[0]
        Wx = np.dot(sqrt_weights_matrix, x_matrix)
        Wa = np.dot(sqrt_weights_matrix, a_matrix)
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

def Autoencoder(y_data, subspace_dim):
    # Reduce the dimensionality of y_data
    AE = {}
    AE['y_data_n_subspace'], AE['PC'] = dimReduce_fast(y_data, subspace_dim)
    
    # Ensure subspace has a significant spread
    while np.min(np.max(AE['y_data_n_subspace'], axis=1) - np.min(AE['y_data_n_subspace'], axis=1)) < 1e-3:
        subspace_dim -= 1
        AE['y_data_n_subspace'], AE['PC'] = dimReduce_fast(y_data, subspace_dim)
        print(f'reduced subspace dim = {subspace_dim}.')

    # Compute weights matrix
    weights_matrix = np.linalg.inv(np.cov(AE['y_data_n_subspace']))
    
    AE['kerneltype'] = 'Gaussian'
    Kxx = KxxMatrix(AE['y_data_n_subspace'], weights_matrix, AE['kerneltype'])
    
    # Perform kernel regularized least squares
    AE['B'] = kernel_regularized_least_squares(Kxx, y_data)
    
    AE['y_data'] = y_data
    AE['sqrt_weights_matrix'] = sqrtm(weights_matrix)
    
    _, distance = AutoencoderFiltering(y_data, AE)
    AE['modeling_error'] = 1 - np.exp(-(1 / y_data.shape[0]) * np.max(distance))
    
    return AE

def AutoencoderFiltering(y_data_new, AE):
    x_data = np.dot(AE['PC'].T, y_data_new)  # Project y_data_new onto AE's principal components
    N = x_data.shape[1]

    if N > 1000:
        hat_y_data = np.zeros((AE['y_data'].shape[0], N))
        N_loops = int(np.ceil(N / 1000))
        
        for loop in range(N_loops):
            ini_ind = 1000 * loop
            fin_ind = min(N, 1000 * (loop + 1))
            
            Kxa = KxaMatrix(x_data[:, ini_ind:fin_ind], AE['y_data_n_subspace'], AE['sqrt_weights_matrix'], AE['kerneltype'])
            weights = np.dot(Kxa, AE['B'])
            sum_weights = np.sum(weights, axis=1, keepdims=True)
            weights = weights / sum_weights
            
            hat_y_data[:, ini_ind:fin_ind] = np.dot(AE['y_data'], weights.T)
    else:
        Kxa = KxaMatrix(x_data, AE['y_data_n_subspace'], AE['sqrt_weights_matrix'], AE['kerneltype'])
        weights = np.dot(Kxa, AE['B'])
        sum_weights = np.sum(weights, axis=1, keepdims=True)
        weights = weights / sum_weights
        
        hat_y_data = np.dot(AE['y_data'], weights.T)
    
    distance = np.sqrt(np.sum((y_data_new - hat_y_data)**2, axis=0))
    
    return hat_y_data, distance



def predictionClassifier(y_data_new, CLF):
    """
    Predict the labels for new data points using a classifier constructed with autoencoders.

    Parameters:
    - y_data_new (numpy.ndarray): New data points to classify.
    - CLF (dict): Classifier model containing autoencoder arrays and labels.

    Returns:
    - min_distance (numpy.ndarray): Minimum distance of each data point to any of the autoencoder models.
    - labels_arr (numpy.ndarray): Predicted labels for each data point.
    """
    C = len(CLF['AE_arr_arr'])  # Number of unique classes
    N = y_data_new.shape[1]  # Number of data points
    distance_matrix = np.zeros((C, N))  # Initialize a distance matrix

    # Compute the distance of each data point to each set of autoencoders (one set per class)
    for i in range(C):
        distances = combineMultipleAutoencoders(y_data_new, CLF['AE_arr_arr'][i])
        distance_matrix[i, :] = distances

    # Find the minimum distance for each data point across all classes
    min_distance = np.min(distance_matrix, axis=0)
    labels_arr = np.zeros(N, dtype=int)  # Initialize the label array

    # Assign labels based on which class had the minimum distance for each data point
    for i in range(C):
        labels_arr[distance_matrix[i, :] == min_distance] = CLF['labels'][i]

    return min_distance, labels_arr



def combineMultipleAutoencoders(y_data_new, AE_arr):
    """
    Combine multiple autoencoders to compute the distance of new data points to the closest autoencoder model.

    Parameters:
    - y_data_new (numpy.ndarray): New data points to be evaluated.
    - AE_arr (list): A list of autoencoder models.

    Returns:
    - distance (numpy.ndarray): Array containing the minimum distance of each data point to any of the autoencoder models.
    """
    N = y_data_new.shape[1]  # Number of data points
    Q = len(AE_arr)  # Number of autoencoders
    distance_matrix = np.zeros((Q, N))  # Initialize a distance matrix

    # Compute the distance of each data point to each autoencoder
    for i in range(Q):
        _, distances = AutoencoderFiltering(y_data_new, AE_arr[i])
        distance_matrix[i, :] = distances

    # Find the minimum distance across all autoencoders for each data point
    distance_min = np.min(distance_matrix, axis=0)
    distance = np.full((1, N), np.inf)  # Initialize the distance array with infinity

    # Assign the minimum distances to the distance array
    for i in range(Q):
        ind = np.where(distance_matrix[i, :] == distance_min)
        distance[0, ind] = distance_matrix[i, ind]

    return distance



def combineMultipleClassifiers(distance_arr, labels_arr_arr):
    """
    Combine multiple classifiers to determine the minimum distance and corresponding labels for data points.

    Parameters:
    - distance_arr (list of numpy.ndarray): List of arrays, each containing distances from a different classifier.
    - labels_arr_arr (list of numpy.ndarray): List of arrays, each containing predicted labels from a different classifier.

    Returns:
    - min_distance (numpy.ndarray): Array of minimum distances for each data point across all classifiers.
    - labels_arr (numpy.ndarray): Array of labels corresponding to the minimum distance for each data point.
    """
    Q = len(distance_arr)  # Number of classifiers
    N = distance_arr[0].shape[1]  # Number of data points

    min_distance = np.full(N, np.inf)  # Initialize with infinity
    labels_arr = np.zeros(N, dtype=int)  # Initialize label array

    # Loop through each classifier's results
    for i in range(Q):
        # Update min_distance with the element-wise minimum
        min_distance = np.minimum(min_distance, distance_arr[i])
        
        # Find indices where the current classifier's distance is the new minimum
        ind = np.where(distance_arr[i] == min_distance)[0]

        # Update labels for these indices
        labels_arr[ind] = labels_arr_arr[i][ind]

    return min_distance, labels_arr