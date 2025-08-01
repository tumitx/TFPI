function U_primary = update_primary_mode(X_tensor, U_factors, primary_idx, priv_indices, model_params, params)
% Update primary mode using LUPI with max-margin constraint
tensor_size = size(X_tensor);
D = size(U_factors{primary_idx}, 2);

% Variational EM for primary mode
[lambda_s, nu_s] = e_step_primary(X_tensor, U_factors, primary_idx, model_params, params);

% M-step: Update factor matrix with max-margin constraint
U_primary = m_step_primary(X_tensor, U_factors, primary_idx, lambda_s, nu_s, model_params, params);
end