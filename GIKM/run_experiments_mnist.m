function [] = run_experiments_mnist()
pkg load statistics
load(fullfile(strrep(pwd,'GIKM','Datasets'),'MNIST','mnist_all.mat'),'train0','train1','train2','train3',...
    'train4','train5','train6',...
    'train7','train8','train9','test0',...
    'test1','test2','test3',...
    'test4','test5','test6',...
    'test7','test8','test9');
y_data_trn = double([train0' train1' train2' train3' train4' train5' train6' train7' train8' train9'])/255;
y_data_trn = tanh(y_data_trn);
labels_trn = [1*ones(1,size(train0,1)) 2*ones(1,size(train1,1)) 3*ones(1,size(train2,1))...
    4*ones(1,size(train3,1)) 5*ones(1,size(train4,1)) 6*ones(1,size(train5,1))...
    7*ones(1,size(train6,1)) 8*ones(1,size(train7,1)) 9*ones(1,size(train8,1))...
    10*ones(1,size(train9,1))];
y_data_test = double([test0' test1' test2' test3' test4' test5' test6' test7' test8' test9'])/255;
y_data_test = tanh(y_data_test);
labels_test = [1*ones(1,size(test0,1)) 2*ones(1,size(test1,1)) 3*ones(1,size(test2,1))...
    4*ones(1,size(test3,1)) 5*ones(1,size(test4,1)) 6*ones(1,size(test5,1))...
    7*ones(1,size(test6,1)) 8*ones(1,size(test7,1)) 9*ones(1,size(test8,1))...
    10*ones(1,size(test9,1))];
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
save('run_experiments_mnist.mat','acc','max_modeling_error');
return





