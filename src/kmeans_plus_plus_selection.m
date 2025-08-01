function anchor_indices = kmeans_plus_plus_selection(X, k)
% K-means++ initialization for anchor selection
[n, ~] = size(X);
anchor_indices = zeros(k, 1);

% Choose first anchor randomly
anchor_indices(1) = randi(n);

for i = 2:k
    % Compute distances to nearest chosen anchor
    distances = inf(n, 1);
    for j = 1:n
        for prev_anchor = 1:(i-1)
            dist = sum((X(j, :) - X(anchor_indices(prev_anchor), :)).^2);
            distances(j) = min(distances(j), dist);
        end
    end
    
    % Choose next anchor with probability proportional to squared distance
    probabilities = distances / sum(distances);
    cumprob = cumsum(probabilities);
    r = rand();
    anchor_indices(i) = find(cumprob >= r, 1);
end
end