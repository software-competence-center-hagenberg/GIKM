function [AE] = Autoencoder(y_data,subspace_dim)
[AE.y_data_n_subspace,AE.PC] = dimReduce(y_data,subspace_dim);
while min((max(AE.y_data_n_subspace,[],2) - min(AE.y_data_n_subspace,[],2))) < 1e-3
    subspace_dim = subspace_dim-1;
    [AE.y_data_n_subspace,AE.PC] = dimReduce(y_data,subspace_dim);
    fprintf('reduced subspace dim = %i.\n',subspace_dim);
end
weights_matrix =  inv(cov(AE.y_data_n_subspace'));
AE.kerneltype = 'Gaussian';
[Kxx] = KxxMatrix(AE.y_data_n_subspace,weights_matrix,AE.kerneltype);
[AE.B] = kernel_regularized_least_squares(Kxx,y_data);
AE.y_data = y_data;
AE.sqrt_weights_matrix = sqrtm(weights_matrix);
[~,distance] = AutoencoderFiltering(y_data,AE);
AE.modeling_error = 1-exp(-(1/size(y_data,1))*max(distance));
return




function [B] = kernel_regularized_least_squares(KernelMatrix,LabelsMatrix)
N = size(LabelsMatrix,2);
tv0 = numel(LabelsMatrix);
tv1 = sum(sum(LabelsMatrix.^2,2))/tv0;
tau = 2*tv1;
e = 0.5*tv1;
lambda = tau + e;
lambda_ini = lambda;
lambda_chg = 1;
itr_count = 0;
while (lambda_chg > 0.01) && (itr_count < 100)
    B = inv(lambda*eye(N) + KernelMatrix);
    temp_matrix = KernelMatrix*B;
    tv = sum(sum((LabelsMatrix-LabelsMatrix*temp_matrix').^2,2));
    lambda = tau + (tv/tv0);
    lambda_chg = abs(lambda-lambda_ini);
    lambda_ini = lambda;
    itr_count = itr_count+1;
end
fprintf('iterations = %i, regularization = %f, N = %i.\n',itr_count,lambda,N);
return
