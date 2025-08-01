function X_tensor = create_tensor_representation(anchor_graphs)
% Create tensor representation from anchor graphs
num_views = length(anchor_graphs);
[n, num_anchors] = size(anchor_graphs{1});

% Stack anchor graphs to form a 3-way tensor
X_tensor = zeros(n, num_anchors, num_views);
for v = 1:num_views
    X_tensor(:, :, v) = anchor_graphs{v};
end
end