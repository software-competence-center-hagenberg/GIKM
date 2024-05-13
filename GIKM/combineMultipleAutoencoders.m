function [distance] = combineMultipleAutoencoders(y_data_new,AE_arr)
N = size(y_data_new,2);
Q = numel(AE_arr);
distance_matrix = zeros(Q,N);
for i = 1:Q
    [~,distance_matrix(i,1:N)] = AutoencoderFiltering(y_data_new,AE_arr{i});
end
distance_min =  min(distance_matrix,[],1);
distance = inf+zeros(1,N);
for i = 1:Q
    ind = find(distance_matrix(i,:) == distance_min);
    distance(1,ind) = distance_matrix(i,ind);
end
return
