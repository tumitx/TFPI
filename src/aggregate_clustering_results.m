function cluster_labels = aggregate_clustering_results(U_factors, K, params)
% Aggregate clustering results from all views
num_views = length(U_factors);
n = size(U_factors{1}, 1);

% Simple aggregation: concatenate all factor matrices and cluster
all_features = [];
for v = 1:num_views
    all_features = [all_features, U_factors{v}];
end

% Apply k-means clustering
cluster_labels = kmeans(all_features, K, 'MaxIter', 300, 'Replicates', 10);
end