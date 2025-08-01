function reconstruction = compute_expected_reconstruction(tensor_slice, U_factors, mode_idx, slice_idx)
% Compute expected reconstruction for a tensor slice
D = size(U_factors{1}, 2);
reconstruction = zeros(D, 1);