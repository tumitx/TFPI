function acc = compute_clustering_accuracy(pred_labels, true_labels)
% Compute clustering accuracy using Hungarian algorithm (simplified)
K = max(true_labels);
confusion_matrix = zeros(K, K);

for i = 1:length(true_labels)
    confusion_matrix(true_labels(i), pred_labels(i)) = confusion_matrix(true_labels(i), pred_labels(i)) + 1;
end

% Find best assignment (simplified - should use Hungarian algorithm)
[~, assignment] = max(confusion_matrix, [], 2);
acc = sum(diag(confusion_matrix(:, assignment))) / length(true_labels);
end