function [Kxa] = KxaMatrix(x_matrix,a_matrix,sqrt_weights_matrix,kerneltype)
switch lower(kerneltype)
    case 'gaussian'
        n = size(x_matrix,1);
        Wx = sqrt_weights_matrix*x_matrix;
        Wa = sqrt_weights_matrix*a_matrix;
        [distMat] = DistanceMatrix(Wx',Wa');
        Kxa = exp(-(0.5/n)*distMat);
end
return