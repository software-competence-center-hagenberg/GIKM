function [] = run_experiments_cifar100_distributed_classes()
rootFolder = fullfile(strrep(pwd,'GIKM','Datasets'),'CIFAR-100');
trainingImages = imageDatastore(fullfile(rootFolder,'TRAIN'),'IncludeSubfolders',true,'LabelSource','foldernames');
testingImages = imageDatastore(fullfile(rootFolder,'TEST'),'IncludeSubfolders',true,'LabelSource','foldernames');
y_data_trn = zeros(2048,numel(trainingImages.Files));
for i = 1:numel(trainingImages.Files)
    [~,~,c] = fileparts(trainingImages.Files{i});
    fprintf('reading feature of file = %s.\n',trainingImages.Files{i});
    load(strrep(trainingImages.Files{i},c,'_resnet50.mat'),'resnet50_features');
    y_data_trn(1:2048,i) = resnet50_features;
end
y_data_trn = tanh(y_data_trn);
trainingLabels = trainingImages.Labels;
classes = (categories(trainingLabels))';
C = numel(classes);
labels_trn = zeros(1,numel(trainingLabels));
for i = 1:C
    labels_trn(1,(trainingLabels)' == classes(i)) = i;
end
y_data_test = zeros(2048,numel(testingImages.Files));
for i = 1:numel(testingImages.Files)
    [~,~,c] = fileparts(testingImages.Files{i});
    fprintf('reading feature of file = %s.\n',testingImages.Files{i});
    load(strrep(testingImages.Files{i},c,'_resnet50.mat'),'resnet50_features');
    y_data_test(1:2048,i) = resnet50_features;
end
y_data_test = tanh(y_data_test);
testingLabels = testingImages.Labels;
labels_test = zeros(1,numel(testingLabels));
for i = 1:C
    labels_test(1,(testingLabels)' == classes(i)) = i;
end
% we simulate a scenarion with the number of clients as equal to number of
% classes
labels = unique(labels_trn);
Q = numel(labels);
min_distance_arr = cell(1,Q);
labels_arr_arr = cell(1,Q);
max_modeling_error = 0;
for i = 1:Q
    ind = find(labels_trn==labels(i));
    CLF = Classifier(y_data_trn(:,ind),...
        labels_trn(:,ind),...
        20,1000);
    [min_distance_arr{i},labels_arr_arr{i}] = predictionClassifier(y_data_test,CLF);
    max_modeling_error = max(max_modeling_error,CLF.max_modeling_error);
end
[~,hat_labels_test] = combineMultipleClassifiers(min_distance_arr,labels_arr_arr);
acc = mean(hat_labels_test==labels_test);
fprintf('global accuracy = %f, maximum modeling error = %f.\n',acc,max_modeling_error);
save('run_experiments_cifar100_distributed_classes.mat','acc','max_modeling_error');
return