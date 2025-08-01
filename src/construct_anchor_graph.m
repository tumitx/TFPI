function anchor_graph = construct_anchor_graph(X, params)
% Construct anchor graph using DAS (Directly Alternating Sampling)
[n, d] = size(X);
num_anchors = round(n * params.anchor_rate);

% Select anchors using k-means++
anchor_indices = kmeans_plus_plus_selection(X, num_anchors);
anchors = X(anchor_indices, :);

% Compute anchor graph (similarity between data points and anchors)
anchor_graph = zeros(n, num_anchors);
for i = 1:n
    distances = sum((X(i, :) - anchors).^2, 2);
    similarities = exp(-distances / (2 * params.sigma^2));
    
    % Keep only top-k connections for sparsity
    [~, sorted_idx] = sort(similarities, 'descend');
    top_k = min(5, num_anchors);  % Connect to top 5 anchors
    
    anchor_graph(i, sorted_idx(1:top_k)) = similarities(sorted_idx(1:top_k));
    % Normalize
    anchor_graph(i, :) = anchor_graph(i, :) / sum(anchor_graph(i, :));
end
end

