function [y_data_n_subspace,PC] = dimReduce(y_data,n)
N = size(y_data,2);
data = y_data - mean(y_data,2);
covariance = (1/(N-1))*(data*data');
[PC, V] = eig(covariance);
[neg_eig_val,rindices] = sort(-1*diag(V));
ind = find(neg_eig_val < -eps);
n = min(n,ind(end));
PC = PC(:,rindices(1:n));
y_data_n_subspace = PC'*y_data;
return