function [Kxx] = KxxMatrix(x_matrix,weights_matrix,kerneltype)
% x_matrix is nXN
% weights_f is nx1
switch lower(kerneltype)
    case 'gaussian'
        [n,N] = size(x_matrix);
        Kxx = zeros(N,N);
        for i = 1:N-1
            for j = (i+1):N
                del = x_matrix(:,i)-x_matrix(:,j);
                Kxx(i,j) = del'*weights_matrix*del;
                Kxx(j,i) = Kxx(i,j);
            end
        end
        Kxx = exp(-(0.5/n)*Kxx);
end
return