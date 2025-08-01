function model_params = update_model_parameters(X_tensor, U_factors, model_params, params)
% Update model parameters (mu, Sigma, tau^2, delta^2)
num_modes = length(U_factors);

for i = 1:num_modes
    % Update mu_i (mean)
    model_params.mu{i} = mean(U_factors{i}, 1)';
    
    % Update Sigma_i (covariance)
    centered = U_factors{i} - model_params.mu{i}';
    model_params.Sigma{i} = (centered' * centered) / size(U_factors{i}, 1) + 1e-6 * eye(size(centered, 2));
end

% Update noise parameters
reconstruction_error = compute_total_reconstruction_error(X_tensor, U_factors);
num_elements = numel(X_tensor);
model_params.tau_squared = reconstruction_error / num_elements;
model_params.delta_squared = 1.0;  % Simplified update
end