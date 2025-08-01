function [lambda_s, nu_s] = e_step_primary(X_tensor, U_factors, primary_idx, model_params, params)
% E-step for primary mode
tensor_size = size(X_tensor);
D = size(U_factors{primary_idx}, 2);
I_s = tensor_size(primary_idx);

% Initialize variational parameters
lambda_s = model_params.mu{primary_idx};
nu_s = diag(model_params.Sigma{primary_idx});

% Update variational parameters (simplified version)
for iter = 1:10  % Inner iterations
    % Compute reconstruction error terms
    reconstruction_term = compute_reconstruction_term(X_tensor, U_factors, primary_idx);
    
    % Update lambda_s
    Sigma_inv = inv(model_params.Sigma{primary_idx});
    precision_matrix = Sigma_inv + reconstruction_term / model_params.tau_squared;
    lambda_s = precision_matrix \ (Sigma_inv * model_params.mu{primary_idx});
    
    % Update nu_s (diagonal of covariance)
    nu_s = 1 ./ diag(precision_matrix);
end
end