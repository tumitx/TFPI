function [U_factors, model_params] = initialize_parameters(X_tensor, K, params)
% Initialize factor matrices and model parameters
tensor_size = size(X_tensor);
num_modes = length(tensor_size);
D = params.tensor_rank;

% Initialize factor matrices
U_factors = cell(num_modes, 1);
for i = 1:num_modes
    U_factors{i} = randn(tensor_size(i), D) * 0.1;
end

% Initialize model parameters
model_params = struct();
model_params.mu = cell(num_modes, 1);
model_params.Sigma = cell(num_modes, 1);
model_params.tau_squared = 1.0;
model_params.delta_squared = 1.0;

for i = 1:num_modes
    model_params.mu{i} = zeros(D, 1);
    model_params.Sigma{i} = eye(D);
end
end