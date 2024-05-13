function [] = run_experiments_fmnist_2()
rng(090324);
rootdir = fullfile(strrep(pwd,'GIKM','Datasets'),'FashionMNIST');
TrainimageFileName = fullfile(rootdir,'train-images-idx3-ubyte');
TrainlabelFileName = fullfile(rootdir,'train-labels-idx1-ubyte');
TestimageFileName = fullfile(rootdir,'t10k-images-idx3-ubyte');
TestlabelFileName = fullfile(rootdir,'t10k-labels-idx1-ubyte');
[XTrain,YTrain] = processFashionMNISTdata(TrainimageFileName,TrainlabelFileName);
[XTest,YTest] = processFashionMNISTdata(TestimageFileName,TestlabelFileName);
y_data_trn = zeros(size(XTrain,1)*size(XTrain,2),size(XTrain,4));
for i = 1:size(XTrain,4)
    temp = XTrain(:,:,1,i);
    y_data_trn(:,i) = double(temp(:))/255;
end
y_data_trn = tanh(y_data_trn);
labels = unique(YTrain);
labels_trn = zeros(1,size(XTrain,4));
for i = 1:numel(labels)
    ind = find(strcmp(YTrain,labels{i}));
    labels_trn(1,ind) = i;
    fprintf('Number of training samples for class %s = %i.\n',labels{i},numel(ind));
end
y_data_test = zeros(size(XTest,1)*size(XTest,2),size(XTest,4));
for i = 1:size(XTest,4)
    temp = XTest(:,:,1,i);
    y_data_test(:,i) = double(temp(:))/255;
end
y_data_test = tanh(y_data_test);
labels_test = zeros(1,size(XTest,4));
for i = 1:numel(labels)
    ind = find(strcmp(YTest,labels{i}));
    labels_test(1,ind) = i;
    fprintf('Number of test samples for class %s = %i.\n',labels{i},numel(ind));
end
n_clients = 100;
n_experiments = 3;
avg_local_acc_arr = zeros(1,n_experiments);
avg_global_acc_arr = zeros(1,n_experiments);
for k = 1:n_experiments
    [client_id_trn] = divide_data_into_non_iid_label_screw(labels_trn,n_clients,round(0.3*numel(labels)));
    local_acc_arr = zeros(1,n_clients);
    distance_matrix = zeros(numel(labels),size(y_data_test,2))+inf;
    for j = 1:n_clients
        [CLF] = Classifier(y_data_trn(:,client_id_trn==j),labels_trn(:,client_id_trn==j),20,1000);
        classes_client = unique(labels_trn(:,client_id_trn==j));
        test_data_ind = [];
        for i = 1:numel(classes_client)
            test_data_ind = [test_data_ind find(labels_test==classes_client(i))];
        end
        y_data_test_client = y_data_test(:,test_data_ind);
        labels_test_client = labels_test(:,test_data_ind);
        [distance_arr,labels_predicted] = predictionClassifier(y_data_test_client,CLF);
        local_acc_arr(1,j) =  mean(labels_predicted==labels_test_client);
        for i = 1:numel(test_data_ind)
            distance_matrix(labels_predicted(i),test_data_ind(i)) = min(distance_arr(i),distance_matrix(labels_predicted(i),test_data_ind(i)));
        end
    end
    min_distance = min(distance_matrix,[],1);
    hat_labels_test = zeros(1,size(y_data_test,2));
    for i = 1:numel(labels)
        hat_labels_test(1,distance_matrix(i,:)==min_distance) = i;
    end
    global_acc_arr = zeros(1,n_clients);
    for j = 1:n_clients
        classes_client = unique(labels_trn(:,client_id_trn==j));
        test_data_ind = [];
        for i = 1:numel(classes_client)
            test_data_ind = [test_data_ind find(labels_test==classes_client(i))];
        end
        global_acc_arr(1,j) =  mean(hat_labels_test(:,test_data_ind)==labels_test(:,test_data_ind));
    end
    avg_local_acc_arr(1,k) = mean(local_acc_arr);
    avg_global_acc_arr(1,k) = mean(global_acc_arr);
end
mean_local_acc_30 = mean(avg_local_acc_arr);
std_local_acc_30 = std(avg_local_acc_arr);
mean_global_acc_30 = mean(avg_global_acc_arr);
std_global_acc_30 = std(avg_global_acc_arr);
save('run_experiments_fmnist_2.mat','mean_local_acc_30','std_local_acc_30',...
    'mean_global_acc_30','std_global_acc_30');
fprintf('local accuracy thirty percentage = %f, std = %f.\n',mean_local_acc_30,std_local_acc_30);
fprintf('global accuracy thirty percentage = %f, std = %f.\n',mean_global_acc_30,std_global_acc_30);
return
