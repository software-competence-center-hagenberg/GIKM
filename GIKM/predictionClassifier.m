function [min_distance,labels_arr] = predictionClassifier(y_data_new,CLF)
C = numel(CLF.AE_arr_arr);
N = size(y_data_new,2);
distance_matrix = zeros(C,N);
for i = 1:C
    [distance_matrix(i,1:N)] = combineMultipleAutoencoders(y_data_new,CLF.AE_arr_arr{i});
end
min_distance = min(distance_matrix,[],1);
labels_arr = zeros(1,N);
for i = 1:C
    labels_arr(1,distance_matrix(i,:)==min_distance) = CLF.labels(i);
end
return
