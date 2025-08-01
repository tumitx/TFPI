function obj_value = compute_objective(X_tensor, U_factors, model_params, params)
% Compute the objective function value
reconstruction_error = compute_total_reconstruction_error(X_tensor, U_factors);
regularization_term = compute_regularization_term(U_factors, model_params);

obj_value = reconstruction_error + params.C * regularization_term;
end