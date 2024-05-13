function [distMat] = DistanceMatrix(A,B)
[numA,d] = size(A);
numB = size(B,1);
if size(B,2) == d
    helpA = zeros(numA,3*d);
    helpB = zeros(numB,3*d);
    for idx = 1:d
        helpA(:,3*idx-2:3*idx) = [ones(numA,1), -2*A(:,idx), A(:,idx).^2 ];
        helpB(:,3*idx-2:3*idx) = [B(:,idx).^2 ,    B(:,idx), ones(numB,1)];
    end
    distMat = helpA * helpB';
else
    distMat = [];
end
return
