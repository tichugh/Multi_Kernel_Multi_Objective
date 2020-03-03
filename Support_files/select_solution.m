function [theta,f_vector] = select_solution(solutions,X,Y,X_test,Y_test,kernels,CS_kernel)

cluster_size = 10;
    
if size(solutions,1)>cluster_size
   idx = kmeans(solutions(:,end-1:end),cluster_size);
end
s_index = zeros(cluster_size,1);
for i = 1:cluster_size
    r = find(idx==i);
    s_index(i) = r(1);
end

solutions = solutions(s_index,:);


mean_Y = mean(Y);
Y = Y - mean_Y;
Y_test = Y_test - mean_Y;

PS = solutions(:,1:end-2);
NN = size(X_test,1); 
n_models = size(PS,1);

msll = zeros(1,n_models);
y_pred = zeros(NN,n_models);

std_dev = zeros(NN,n_models);

for i = 1:n_models

par = PS(i,:);
N = size(X,1);

K = Cov_matrix_calculations(X,par,kernels,CS_kernel) + 1e-8 * eye(N,N); % N x N covariance matrix

K_star1 = [];
for ii = 1:N
    for j = 1:size(X_test,1)
%         K_star1(ii,j) = kernel_with_noise_multi_kernel(X(ii,:),X_test(j,:),par,kernels);
         K_star1(ii,j) = multi_kernel_calculation(X(ii,:),X_test(j,:),par,kernels,CS_kernel);
    end
end

% K_star2 = Cov_Mat_with_noise_multi_kernel(X_test,par,kernels) + 0.00001 * eye(NN,NN);
K_star2 = Cov_matrix_calculations(X_test,par,kernels,CS_kernel) + 1e-8 * eye(NN,NN);

TT = chol(K,'lower');
K_inv = inv_chol(TT);

m = K_star1'*K_inv*Y; % caluated mean for the test data
Cov = K_star2 - K_star1'*K_inv*K_star1; % covariance for the test data
Var_test = diag(Cov);
ST_test = sqrt(Var_test);
    
y_pred(:,i) = m;
std_dev(:,i) = ST_test;



msll(:,i) = calculate_msll(Y_test,y_pred(:,i),std_dev(:,i),0);
end
msll = real(msll);
[~,index] = min(msll);
sol_index = index;
theta = solutions(sol_index,1:end-2);
f_vector = solutions(sol_index,end-1:end);

end
