function X_recon = reconstruct_tensor(U_factors)
% Reconstruct tensor from factor matrices using CP decomposition
D = size(U_factors{1}, 2);
tensor_size = [size(U_factors{1}, 1), size(U_factors{2}, 1), size(U_factors{3}, 1)];

X_recon = zeros(tensor_size);
for d = 1:D
    outer_product = U_factors{1}(:, d) * U_factors{2}(:, d)' * reshape(U_factors{3}(:, d), [1, 1, length(U_factors{3}(:, d))]);
    X_recon = X_recon + outer_product;
end
end