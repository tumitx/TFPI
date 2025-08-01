function demo_TFPI()
% Generate synthetic multi-view data
rng(42);
n = 200;  % number of samples
K = 3;    % number of clusters
num_views = 3;

% Generate synthetic data
X_views = cell(num_views, 1);
true_labels = repmat(1:K, 1, ceil(n/K));
true_labels = true_labels(1:n);

for v = 1:num_views
    X_views{v} = randn(n, 50 + v*10);  % Different dimensionalities
    % Add some structure based on true clusters
    for k = 1:K
        cluster_mask = (true_labels == k);
        X_views{v}(cluster_mask, :) = X_views{v}(cluster_mask, :) + randn(1, size(X_views{v}, 2)) * 2;
    end
end

% Set parameters
params = struct();
params.anchor_rate = 0.2;
params.C = 0.01;
params.epsilon = 0.01;
params.max_iter = 50;
params.tensor_rank = 10;

% Run TFPI clustering
fprintf('Running TFPI Multi-View Clustering...\n');
[cluster_labels, U_factors, final_obj] = TFPI_MVC(X_views, K, params);

% Evaluate clustering performance
acc = compute_clustering_accuracy(cluster_labels, true_labels);
nmi = compute_nmi(cluster_labels, true_labels);

fprintf('Clustering Results:\n');
fprintf('Accuracy: %.4f\n', acc);
fprintf('NMI: %.4f\n', nmi);
fprintf('Final Objective: %.4f\n', final_obj);
end