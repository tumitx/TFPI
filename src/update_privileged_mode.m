function U_priv = update_privileged_mode(X_tensor, U_factors, priv_idx, model_params, params)
% Update privileged mode using standard tensor factorization
tensor_size = size(X_tensor);
D = size(U_factors{priv_idx}, 2);

% Variational update for privileged mode
U_priv = U_factors{priv_idx};

% Compute reconstruction error
for i = 1:tensor_size(priv_idx)
    % Extract tensor slice
    if priv_idx == 1
        tensor_slice = squeeze(X_tensor(i, :, :));
    elseif priv_idx == 2
        tensor_slice = squeeze(X_tensor(:, i, :));
    else
        tensor_slice = squeeze(X_tensor(:, :, i));
    end
    
    % Update factor
    reconstruction = compute_expected_reconstruction(tensor_slice, U_factors, priv_idx, i);
    regularization = model_params.Sigma{priv_idx} \ (U_priv(i, :)' - model_params.mu{priv_idx});
    
    step_size = 0.01;
    U_priv(i, :) = U_priv(i, :) + step_size * (reconstruction - regularization)';
end
end