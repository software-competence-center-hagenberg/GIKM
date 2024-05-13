function [cluster_labels] = k_means_clustering(data,no_of_clusters)
stream = RandStream('mlfg6331_64');  % Random number stream
options = statset('UseSubstreams',1,'Streams',stream);
[idx,~] = kmeans(data',no_of_clusters,'Options',options,'MaxIter',10000,...
    'Display','off','Replicates',10);
cluster_labels = idx';
return