function [] = run_experiments_office_caltech256_2()
N_exp = 20;
acc_matrix = zeros(12,N_exp);
acc_matrix_ref = zeros(12,N_exp);
for j = 1:N_exp
    [acc_matrix(1,j),acc_matrix_ref(1,j)] = Amazon2Caltech(j,'vgg16');
    [acc_matrix(2,j),acc_matrix_ref(2,j)] = Amazon2Dslr(j,'vgg16');
    [acc_matrix(3,j),acc_matrix_ref(3,j)] = Amazon2Webcam(j,'vgg16');
    [acc_matrix(4,j),acc_matrix_ref(4,j)] = Caltech2Amazon(j,'vgg16');
    [acc_matrix(5,j),acc_matrix_ref(5,j)] = Caltech2Dslr(j,'vgg16');
    [acc_matrix(6,j),acc_matrix_ref(6,j)] = Caltech2Webcam(j,'vgg16');
    [acc_matrix(7,j),acc_matrix_ref(7,j)] = Dslr2Amazon(j,'vgg16');
    [acc_matrix(8,j),acc_matrix_ref(8,j)] = Dslr2Caltech(j,'vgg16');
    [acc_matrix(9,j),acc_matrix_ref(9,j)] = Dslr2Webcam(j,'vgg16');
    [acc_matrix(10,j),acc_matrix_ref(10,j)] = Webcam2Amazon(j,'vgg16');
    [acc_matrix(11,j),acc_matrix_ref(11,j)] = Webcam2Caltech(j,'vgg16');
    [acc_matrix(12,j),acc_matrix_ref(12,j)] = Webcam2Dslr(j,'vgg16');
end
save('run_experiments_office_caltech256_2.mat','acc_matrix','acc_matrix_ref');
return
