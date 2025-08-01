function params = set_default_params(params)
% Set default parameters for TFPI
if ~isfield(params, 'anchor_rate'), params.anchor_rate = 0.1; end
if ~isfield(params, 'C'), params.C = 0.01; end  % Regularization parameter
if ~isfield(params, 'epsilon'), params.epsilon = 0.01; end  % Epsilon for SVM
if ~isfield(params, 'max_iter'), params.max_iter = 100; end
if ~isfield(params, 'tol'), params.tol = 1e-6; end
if ~isfield(params, 'tensor_rank'), params.tensor_rank = 10; end
if ~isfield(params, 'sigma'), params.sigma = 1.0; end  % RBF kernel parameter
end