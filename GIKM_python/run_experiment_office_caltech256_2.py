from GIKM_python.func_office_caltech import get_data
from func import Classifier, predictionClassifier, combineMultipleClassifiers
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

def run_experiments_office_caltech(random_state, feature_type):
    np.random.seed(random_state)
    # Load the source and target data
    y_data_source_trn, labels_source_trn, _, _ = get_data('amazon_10', 20, feature_type)
    y_data_source_trn = np.tanh(y_data_source_trn)
    
    y_data_target_trn, labels_target_trn, y_data_target_test, labels_target_test = get_data('caltech256_10', 2, feature_type)
    y_data_target_trn = np.tanh(y_data_target_trn)
    y_data_target_test = np.tanh(y_data_target_test)

    # Train classifier on source domain
    clf = Classifier(y_data_source_trn, labels_source_trn, 19, 1000)
    min_distance_1, labels_1 = predictionClassifier(y_data_target_test, clf)

    # Train classifier on target domain
    clf = Classifier(y_data_target_trn, labels_target_trn, 2, 1000)
    min_distance_2, labels_2 = predictionClassifier(y_data_target_test, clf)

    # Combine classifier results
    min_distance_arr = [min_distance_1, min_distance_2]
    labels_arr_arr = [labels_1, labels_2]
    _, hat_labels_test = combineMultipleClassifiers(min_distance_arr, labels_arr_arr)

    # Calculate transfer learning accuracy
    acc = np.mean(hat_labels_test == labels_target_test)
    print(f'transfer learning: test data accuracy = {acc:.6f}')

    # Train a reference SVM classifier on target domain and test
    classifier = SVC()
    classifier.fit(y_data_target_trn.T, labels_target_trn)
    predicted_labels = classifier.predict(y_data_target_test.T)
    
    # Calculate reference accuracy
    acc_ref = accuracy_score(labels_target_test, predicted_labels)
    print(f'SVM: test data accuracy = {acc_ref:.6f}')
    return 0



def main():
    run_experiments_office_caltech(4232,'resnet50')


if __name__ == "__main__":
    main()