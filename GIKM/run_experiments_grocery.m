function [] = run_experiments_grocery()
imagesFolder = fullfile(strrep(pwd,'GIKM','Datasets'),'FreiburgGrocery');
imds = imageDatastore(imagesFolder,'IncludeSubfolders',true,'LabelSource','foldernames');
classes = (categories(imds.Labels))';
T = readtable('trainGrocery.txt','Delimiter',' ');
[~,~,trainingIdx] = intersect(fullfile(imagesFolder,T.Var1),imds.Files,'stable');
T = readtable('testGrocery.txt','Delimiter',' ');
[~,~,testingIdx] = intersect(fullfile(imagesFolder,T.Var1),imds.Files,'stable');
clear T;
y_data_trn = zeros(2048,numel(trainingIdx));
labels_trn = zeros(1,numel(trainingIdx));
for i = 1:numel(trainingIdx)
    [~,~,c] = fileparts(imds.Files{trainingIdx(i)});
    fprintf('reading feature of file = %s.\n',imds.Files{trainingIdx(i)});
    load(strrep(imds.Files{trainingIdx(i)},c,'_resnet50.mat'),'resnet50_features');
    y_data_trn(1:2048,i) = resnet50_features;
    labels_trn(1,i) = find(imds.Labels(trainingIdx(i))==classes);
end
y_data_trn = tanh(y_data_trn);
y_data_test = zeros(2048,numel(testingIdx));
labels_test = zeros(1,numel(testingIdx));
for i = 1:numel(testingIdx)
    [~,~,c] = fileparts(imds.Files{testingIdx(i)});
    fprintf('reading feature of file = %s.\n',imds.Files{testingIdx(i)});
    load(strrep(imds.Files{testingIdx(i)},c,'_resnet50.mat'),'resnet50_features');
    y_data_test(1:2048,i) = resnet50_features;
    labels_test(1,i) = find(imds.Labels(testingIdx(i))==classes);
end
y_data_test = tanh(y_data_test);
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
save('run_experiments_grocery.mat','acc','max_modeling_error');
return

