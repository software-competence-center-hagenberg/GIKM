function [acc,acc_ref] = Webcam2Caltech(random_state,feature_type)
rng(random_state);
min_distance_arr = cell(1,2);
labels_arr_arr = cell(1,2);
[y_data_source_trn,labels_source_trn,~,~] = getData('webcam_10',8,feature_type);
y_data_source_trn = tanh(y_data_source_trn);
[y_data_target_trn,labels_target_trn,y_data_target_test,labels_target_test] = getData('caltech256_10',3,feature_type);
y_data_target_trn = tanh(y_data_target_trn);
y_data_target_test = tanh(y_data_target_test);
CLF = Classifier(y_data_source_trn,...
    labels_source_trn,...
    7,1000);
[min_distance_arr{1},labels_arr_arr{1}] = predictionClassifier(y_data_target_test,CLF);
CLF = Classifier(y_data_target_trn,...
    labels_target_trn,...
    2,1000);
[min_distance_arr{2},labels_arr_arr{2}] = predictionClassifier(y_data_target_test,CLF);
[~,hat_labels_test] = combineMultipleClassifiers(min_distance_arr,labels_arr_arr);
acc = mean(hat_labels_test==labels_target_test);
fprintf('transfer learning: test data accuracy = %f.\n',acc);
classifier = fitcecoc(y_data_target_trn',labels_target_trn');
predictedLabels = predict(classifier,y_data_target_test');
acc_ref = mean(predictedLabels == labels_target_test');
fprintf('SVM: test data accuracy = %f.\n',acc_ref);
return







