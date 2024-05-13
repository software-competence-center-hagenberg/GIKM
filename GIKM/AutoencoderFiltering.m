function [hat_y_data,distance] = AutoencoderFiltering(y_data_new,AE)
x_data = (AE.PC)'*y_data_new;
N = size(x_data,2);
if N > 1000
    hat_y_data = zeros(size(AE.y_data,1),N);
    N_loops = ceil(N/1000);
    for loop = 1:N_loops
        ini_ind = 1000*loop-999;
        fin_ind = min(N,1000*loop);
        [Kxa] = KxaMatrix(x_data(:,ini_ind:fin_ind),AE.y_data_n_subspace,AE.sqrt_weights_matrix,AE.kerneltype);
        weights = Kxa*AE.B;
        sum_weights = sum(weights,2);
        weights = weights./repmat(sum_weights,1,size(AE.B,2));
        hat_y_data(:,ini_ind:fin_ind) = AE.y_data*weights';
    end
else
    [Kxa] = KxaMatrix(x_data,AE.y_data_n_subspace,AE.sqrt_weights_matrix,AE.kerneltype);
    weights = Kxa*AE.B;
    sum_weights = sum(weights,2);
    weights = weights./repmat(sum_weights,1,size(AE.B,2));
    hat_y_data = AE.y_data*weights';
end
distance = (sum((y_data_new-hat_y_data).^2,1)).^0.5;
return
