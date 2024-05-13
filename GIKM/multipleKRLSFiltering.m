function [hat_y_data,distance] = multipleKRLSFiltering(y_data_new,AE_arr)
Q = numel(AE_arr);
N = size(y_data_new,2);
if Q > 1
    distance_matrix = zeros(Q,N);
    hat_y_data_cell = cell(1,Q);
    for i = 1:Q
        [hat_y_data_cell{i},distance_matrix(i,1:N)] = KRLSFiltering(y_data_new,AE_arr{i});
    end
    distance =  min(distance_matrix,[],1);
    hat_y_data = 0*hat_y_data_cell{1};
    for i = 1:Q
        ind = find(distance_matrix(i,:) == distance);
        hat_y_data(:,ind) = hat_y_data_cell{i}(:,ind);
    end
else
    [hat_y_data,distance] = KRLSFiltering(y_data_new,AE_arr{1});
end
return