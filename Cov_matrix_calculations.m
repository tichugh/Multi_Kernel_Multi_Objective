function f = Cov_matrix_calculations(X,par,kernels,CS_kernel)

m = size(X,1);
k = zeros(m,m);


Z = ones(m,m);
LL = tril(Z);

for i = 1:m
    c_l = sum(LL(i,:));
    xi = X(i,:);
    for j = 1:c_l
        k(i,j) = multi_kernel_calculation(xi,X(j,:),par,kernels,CS_kernel);
    end
end


k = k'+triu(k',1)';

f = k;