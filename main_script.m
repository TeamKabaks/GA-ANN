% Main script to run GA optimization on neural network

fprintf('Hello');
% Load plant data
[X, y] = loadPlantData('Plants/plants.data');
fprintf('Data loaded');
% Define network architecture
input_layer_size = size(X, 2);    % Number of features (locations)
hidden_layer_size = 25;           % Number of hidden units
num_labels = length(unique(y));   % Number of classes (genera)
lambda = 0.01;                    % Regularization parameter

% Split data into training and testing sets
rng(1);  % For reproducibility
n = length(y);
idx = randperm(n);  % Shuffle indices
splitIdx = round(0.7 * n);  % 70% training, 30% testing

trainIdx = idx(1:splitIdx);
testIdx = idx(splitIdx+1:end);

X_train = X(trainIdx, :);
y_train = y(trainIdx);
X_test = X(testIdx, :);
y_test = y(testIdx);

% Display dataset sizes for debugging
fprintf('Original dataset: %d samples, %d features\n', size(X, 1), size(X, 2));
fprintf('Training set: %d samples\n', size(X_train, 1));
fprintf('Testing set: %d samples\n', size(X_test, 1));

% Configure GA parameters
ga_params = struct();
ga_params.populationSize = 50;
ga_params.maxGenerations = 100;
ga_params.crossoverRate = 0.8;
ga_params.mutationRate = 0.03;
ga_params.eliteCount = 2;
ga_params.tournamentSize = 3;

% Run GA optimization
fprintf('Starting Genetic Algorithm optimization...\n');
tic;
[best_nn_params, best_cost] = ga_optimize_nn(input_layer_size, hidden_layer_size, ...
                                       num_labels, X_train, y_train, lambda, ga_params);
time_taken = toc;
fprintf('GA optimization completed in %.2f seconds. Best cost: %.6f\n', time_taken, best_cost);

% Reshape best parameters into weight matrices
Theta1 = reshape(best_nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1));
Theta2 = reshape(best_nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                num_labels, (hidden_layer_size + 1));

% Evaluate on test set
pred = predict(Theta1, Theta2, X_test);
accuracy = mean(double(pred == y_test)) * 100;
fprintf('Test set accuracy: %.2f%%\n', accuracy);

% Create confusion matrix
confMat = zeros(num_labels, num_labels);
for i = 1:length(y_test)
    confMat(y_test(i), pred(i)) = confMat(y_test(i), pred(i)) + 1;
end

% Display confusion matrix
figure;
imagesc(confMat);
colorbar;
title('Confusion Matrix');
xlabel('Predicted');
ylabel('Actual');
