function [min_distance,labels_arr] = combineMultipleClassifiers(distance_arr,labels_arr_arr)
Q = numel(distance_arr);
N = size(distance_arr{1},2);
min_distance = zeros(1,N)+inf;
labels_arr = zeros(1,N);
for i = 1:Q
    min_distance = min(min_distance,distance_arr{i});
    ind = find(distance_arr{i}==min_distance);
    labels_arr(1,ind) = labels_arr_arr{i}(1,ind);
end
return


