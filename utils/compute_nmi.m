function nmi_value = compute_nmi(pred_labels, true_labels)
% Compute Normalized Mutual Information (simplified version)
n = length(pred_labels);
K_true = max(true_labels);
K_pred = max(pred_labels);

% Compute marginal probabilities
p_true = histcounts(true_labels, 1:K_true+1) / n;
p_pred = histcounts(pred_labels, 1:K_pred+1) / n;

% Compute joint probabilities
joint_prob = zeros(K_true, K_pred);
for i = 1:n
    joint_prob(true_labels(i), pred_labels(i)) = joint_prob(true_labels(i), pred_labels(i)) + 1;
end
joint_prob = joint_prob / n;

% Compute mutual information
mi = 0;
for i = 1:K_true
    for j = 1:K_pred
        if joint_prob(i, j) > 0
            mi = mi + joint_prob(i, j) * log(joint_prob(i, j) / (p_true(i) * p_pred(j)));
        end
    end
end

% Compute entropies
h_true = -sum(p_true .* log(p_true + eps));
h_pred = -sum(p_pred .* log(p_pred + eps));

% Normalized mutual information
nmi_value = 2 * mi / (h_true + h_pred);
end