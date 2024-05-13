function [] = analyzeTransferLearningResults()
a_matrix = zeros(12,2);
load('run_experiments_office_caltech256.mat','acc_matrix');
a_matrix(:,1) = round(100*mean(acc_matrix,2),1);
load('run_experiments_office_caltech256_2.mat','acc_matrix');
a_matrix(:,2) = round(100*mean(acc_matrix,2),1);
b_matrix = zeros(12,10);
b_matrix(1,:) = [80.6 83.3 78.1 78.7 75.5 41.5 43.6 35.3 36.4 31.0];
b_matrix(2,:) = [91.2 87.7 86.9 77.1 87.1 60.2 49.8 60.4 56.7 55.1];
b_matrix(3,:) = [89.5 90.7 91.2 82.5 87.9 72.4 59.7 68.7 64.6 57.4];
b_matrix(4,:) = [91.5 89.7 88.0 85.9 86.2 54.8 55.1 50.9 49.4 43.8];
b_matrix(5,:) = [91.6 86.9 86.3 77.9 87.0 61.5 56.2 59.8 56.5 55.6];
b_matrix(6,:) = [91.6 91.4 89.7 82.8 86.0 71.1 62.9 66.3 63.8 58.1];
b_matrix(7,:) = [90.7 88.7 88.1 83.6 85.9 54.4 55.0 50.7 46.9 42.9];
b_matrix(8,:) = [81.4 81.4 77.9 71.8 74.8 40.3 41.0 34.9 34.1 30.9];
b_matrix(9,:) = [88.7 95.5 90.7 86.1 86.9 83.2 80.1 68.5 74.1 60.5];
b_matrix(10,:) = [92.0 88.8 87.4 84.7 85.1 55.0 54.3 51.8 47.7 56.5];
b_matrix(11,:) = [82.3 82.8 78.2 73.6 74.4 37.4 38.6 33.5 32.2 29.0];
b_matrix(12,:) = [89.6 94.5 88.5 85.1 87.3 75.0 70.8 60.7 67.0 56.5];
matrix = [a_matrix b_matrix];
results = round([(mean(matrix,1))' (std(matrix,[],1))'],1);
for i = 1:size(results,1)
    fprintf('%.1f %s %.1f\n',results(i,1),'$\pm$',results(i,2));
end
return