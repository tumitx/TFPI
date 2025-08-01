function reconstruction = compute_expected_reconstruction(tensor_slice, U_factors, mode_idx, slice_idx)
% Compute expected reconstruction for a tensor slice
D = size(U_factors{1}, 2);
reconstruction = zeros(D, 1);

% This is a simplified version - in practice would involve more complex tensor operations
for d = 1:D
    factor_product = 1;
    for j = 1:length(U_factors)
        if j ~= mode_idx
            if j == 1 && mode_idx ~= 1
                factor_product = factor_product * sum(tensor_slice .* U_factors{j}(:, d));
            elseif j == 2 && mode_idx ~= 2
                factor_product = factor_product * sum(tensor_slice .* U_factors{j}(:, d)');
            end
        end
    end
    reconstruction(d) = factor_product;
end
end