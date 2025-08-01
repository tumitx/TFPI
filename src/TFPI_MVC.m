function [cluster_labels, U_factors, final_obj] = TFPI_MVC(X_views, num_clusters, params)
% Multi-View Clustering with Privileged Information based on Probabilistic Tensor Factorization
% 
% Inputs:
%   X_views: Cell array containing multi-view data {X^(1), X^(2), ..., X^(V)}
%   num_clusters: Number of clusters (K)
%   params: Structure containing hyperparameters
%
% Outputs:
%   cluster_labels: Final cluster assignments
%   U_factors: Learned factor matrices
%   final_obj: Final objective value

if nargin < 3
    params = struct();
end

% Set default parameters
params = set_default_params(params);

num_views = length(X_views);
[num_samples, ~] = size(X_views{1});

% Step 1: Construct anchor graphs for each view
fprintf('Constructing anchor graphs...\n');
anchor_graphs = cell(num_views, 1);
for v = 1:num_views
    anchor_graphs{v} = construct_anchor_graph(X_views{v}, params);
end

% Step 2: Create tensor representation
fprintf('Creating tensor representation...\n');
X_tensor = create_tensor_representation(anchor_graphs);

% Step 3: Initialize parameters
fprintf('Initializing parameters...\n');
[U_factors, model_params] = initialize_parameters(X_tensor, num_clusters, params);

% Step 4: Main optimization loop with LUPI
fprintf('Starting TFPI optimization...\n');
obj_values = [];
cluster_results = cell(num_views, 1);

for iter = 1:params.max_iter
    fprintf('Iteration %d/%d\n', iter, params.max_iter);
    
    % For each view as primary
    for primary_view = 1:num_views
        % Set privileged views (all others)
        privileged_views = setdiff(1:num_views, primary_view);
        
        % Update primary mode U^(s) with LUPI
        U_factors{primary_view} = update_primary_mode(X_tensor, U_factors, ...
            primary_view, privileged_views, model_params, params);
        
        % Update privileged modes U^(t)
        for priv_view = privileged_views
            U_factors{priv_view} = update_privileged_mode(X_tensor, U_factors, ...
                priv_view, model_params, params);
        end
        
        % Update model parameters
        model_params = update_model_parameters(X_tensor, U_factors, model_params, params);
    end
    
    % Compute objective value
    current_obj = compute_objective(X_tensor, U_factors, model_params, params);
    obj_values(end+1) = current_obj;
    
    % Check convergence
    if iter > 1 && abs(obj_values(end) - obj_values(end-1)) < params.tol
        fprintf('Converged at iteration %d\n', iter);
        break;
    end
end

% Step 5: Final clustering aggregation
fprintf('Performing final clustering...\n');
cluster_labels = aggregate_clustering_results(U_factors, num_clusters, params);
final_obj = obj_values(end);

fprintf('TFPI clustering completed!\n');
end

































