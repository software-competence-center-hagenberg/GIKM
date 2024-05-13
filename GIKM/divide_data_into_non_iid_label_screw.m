function [client_id_trn] = divide_data_into_non_iid_label_screw(labels_trn,n_clients,n_classes_per_client)
labels = unique(labels_trn);
C = numel(labels);
n_groups_per_class_arr = zeros(1,C);
while (sum(n_groups_per_class_arr==0) > 0)
    tM = randi([min(labels) max(labels)],n_classes_per_client,n_clients);
    for i = 1:C
        n_groups_per_class_arr(1,i) = sum(sum(tM==labels(i),1));
    end
end
ind_cell_cell = cell(1,C);
for i = 1:C
    ind = find(labels_trn==labels(i));
    [group] = divide_data_into_groups(numel(ind),n_groups_per_class_arr(1,i),1);
    ind_cell_cell{i} = cell(1,n_groups_per_class_arr(1,i));
    for j = 1:n_groups_per_class_arr(1,i)
        ind_cell_cell{i}{j} = ind(group==j);
    end
end
client_trn_ind_cell_cell = cell(n_classes_per_client,n_clients);
for i = 1:C
    [a,b] = find(tM==labels(i));
    for j = 1:numel(a)
        client_trn_ind_cell_cell{a(j),b(j)} = ind_cell_cell{i}{j};
    end
end
client_id_trn = zeros(1,numel(labels_trn));
for i = 1:n_clients
    for j = 1:n_classes_per_client
        if numel(client_trn_ind_cell_cell{j,i}) > 0
            client_id_trn(1,client_trn_ind_cell_cell{j,i}) = i;
        end
    end
end
return