
% Run the demo
demo_TFPI();

% Or use with your own data
params = struct('anchor_rate', 0.2, 'C', 0.01, 'max_iter', 100);
[labels, factors, obj] = TFPI_MVC(your_multiview_data, num_clusters, params);
