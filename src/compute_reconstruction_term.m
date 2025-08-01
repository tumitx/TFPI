function reconstruction_term = compute_reconstruction_term(X_tensor, U_factors, primary_idx)
% Compute reconstruction term for variational updates
tensor_size = size(X_tensor);
D = size(U_factors{1}, 2);

% Simplified computation
reconstruction_term = eye(D);  % This should be computed based on tensor contractions
end