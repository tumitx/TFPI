function error = compute_total_reconstruction_error(X_tensor, U_factors)
% Compute total reconstruction error
X_reconstructed = reconstruct_tensor(U_factors);
error = norm(X_tensor(:) - X_reconstructed(:))^2;
end