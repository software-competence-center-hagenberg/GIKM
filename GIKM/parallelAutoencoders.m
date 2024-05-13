function [AE_arr] = parallelAutoencoders(y_data,subspace_dim,Nb)
AE_arr = {};
N = size(y_data,2);
S = round(N/Nb);
if S > 1
    [cluster_labels] = k_means_clustering(y_data,S);
    labels_indices_less = [];
    for i = 1:S
        ind = find(cluster_labels==i);
        if numel(ind) < 2
            labels_indices_less = [labels_indices_less i];
        end
    end
    while ~isempty(labels_indices_less)
        S = S-1;
        [cluster_labels] = k_means_clustering(y_data,S);
        labels_indices_less = [];
        for i = 1:S
            ind = find(cluster_labels==i);
            if numel(ind) < 30
                labels_indices_less = [labels_indices_less i];
            end
        end
    end
    AE_arr = cell(1,S);
    y_data_cell = cell(1,S);
    for i = 1:S
        y_data_cell{i} = y_data(:,cluster_labels==i);
    end
    parfor i = 1:S
        AE_arr{i} = Autoencoder(y_data_cell{i},subspace_dim);
    end
else
    AE_arr{end+1} = Autoencoder(y_data,subspace_dim);
end

return