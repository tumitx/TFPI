function U_primary = m_step_primary(X_tensor, U_factors, primary_idx, lambda_s, nu_s, model_params, params)
% M-step for primary mode with SVM constraint
D = length(lambda_s);
I_s = size(U_factors{primary_idx}, 1);

% Initialize with current estimate
U_primary = U_factors{primary_idx};

% Simplified update (in practice, this would involve solving the SVM problem)
for i = 1:I_s
    % Extract relevant tensor slice
    tensor_slice = squeeze(X_tensor(i, :, :));
    
    % Compute expected reconstruction
    reconstruction = compute_expected_reconstruction(tensor_slice, U_factors, primary_idx, i);
    
    % Update with regularization
    regularization = model_params.Sigma{primary_idx} \ (U_primary(i, :)' - model_params.mu{primary_idx});
    gradient = reconstruction - params.C * regularization;
    
    % Gradient descent step
    step_size = 0.01;
    U_primary(i, :) = U_primary(i, :) + step_size * gradient';
end
end