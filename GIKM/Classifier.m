function [CLF] = Classifier(y_data,label_data,subspace_dim,Nb)
CLF.labels = unique(label_data);
C = numel(CLF.labels);
CLF.AE_arr_arr = cell(1,C);
max_modeling_error = 0;
for i = 1:C
    fprintf('Building an autoencoder for class = %s...\n',num2str(CLF.labels(i)));
    class_data = y_data(:,label_data==CLF.labels(i));
    CLF.AE_arr_arr{i} = parallelAutoencoders(class_data,subspace_dim,Nb);
    for j = 1:numel(CLF.AE_arr_arr{i})
        max_modeling_error = max(max_modeling_error,CLF.AE_arr_arr{i}{j}.modeling_error);
    end
    fprintf('maximum modeling error = %f.\n',max_modeling_error);
end
CLF.max_modeling_error = max_modeling_error;
fprintf('overall maximum modeling error = %f.\n',max_modeling_error);
return

