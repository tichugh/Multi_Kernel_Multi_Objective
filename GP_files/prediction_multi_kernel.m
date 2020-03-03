function [y_pred,std_dev] = prediction_multi_kernel(par,X,Y,X_test,kernels,CS_kernel)

mean_Y = mean(Y);
Y = Y-mean_Y;

N = size(X,1);
% K = Cov_Mat_with_noise_multi_kernel(X,par,kernels) + 0.00001 * eye(N,N); % N x N covariance matrix
K = Cov_matrix_calculations(X,par,kernels,CS_kernel) + 1e-8 * eye(N,N); % N x N covariance matrix

K_star1 = [];
for ii = 1:N
    for j = 1:size(X_test,1)
        K_star1(ii,j) = multi_kernel_calculation(X(ii,:),X_test(j,:),par,kernels,CS_kernel);
    end
end

NN = size(X_test,1);
K_star2 = Cov_matrix_calculations(X_test,par,kernels,CS_kernel) + 1e-8 * eye(NN,NN);

TT = chol(K,'lower');
K_inv = inv_chol(TT);

m = K_star1'*K_inv*Y; % caluated mean for the test data
Cov = K_star2 - K_star1'*K_inv*K_star1; % covariance for the test data
Var_test = diag(Cov);
ST_test = sqrt(Var_test);
    
y_pred = m;
y_pred = y_pred + mean_Y;
std_dev = ST_test;




