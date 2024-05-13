function [y_data_trn,labels_trn,y_data_test,labels_test] = getData(FolderName,N_per_class,feature_type)
imds = imageDatastore(fullfile(strrep(pwd,'GIKM','Datasets'),'office+caltech256',FolderName),'IncludeSubfolders',true,'LabelSource','foldernames');
classes = (categories(imds.Labels))';
[trainingImages, testingImages] = splitEachLabel(imds,N_per_class,'randomized');
N_tr = numel(trainingImages.Files);
y_data_trn = cell(1,N_tr);
str_1 = strcat('_',feature_type,'.mat');
str_2 = strcat(feature_type,'_features');
for i = 1:N_tr
    [~,~,c] = fileparts(trainingImages.Files{i});
    load(strrep(trainingImages.Files{i},c,str_1),str_2);
    y_data_trn{i} = eval(str_2);
end
y_data_trn = cell2mat(y_data_trn);
labels_trn = zeros(1,N_tr);
for i = 1:numel(classes)
    labels_trn(1,(find(trainingImages.Labels==classes(i)))') = i;
end
N_test = numel(testingImages.Files);
y_data_test = cell(1,N_test);
for i = 1:N_test
    [~,~,c] = fileparts(testingImages.Files{i});
    load(strrep(testingImages.Files{i},c,str_1),str_2);
    y_data_test{i} = eval(str_2);
end
y_data_test = cell2mat(y_data_test);
labels_test = zeros(1,N_test);
for i = 1:numel(classes)
    labels_test(1,(find(testingImages.Labels==classes(i)))') = i;
end
return