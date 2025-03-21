% Main script for handwritten digit recognition using GA-optimized NN

% Load digit recognition datasets
[X_train, y_train] = loadDigitData('Original Data\optdigits-orig.tra');
[X_wd, y_wd] = loadDigitData('Original Data\optdigits-orig.wdep');
[X_wi, y_wi] = loadDigitData('Original Data\optdigits-orig.windep');
[X_val, y_val] = loadDigitData('Original Data\optdigits-orig.cv');

fprintf('Data loaded: %d training samples, %d validation samples\n', ...
        size(X_train, 1), size(X_val, 1));


% Define network architecture
input_layer_size = size(X_train, 2);   % Number of features (pixels)
hidden_layer_size = 50;                % Adjust based on complexity
num_labels = 10;                       % 10 digits (0-9)
lambda = 0.1;                          % Regularization parameter

% Configure GA parameters
ga_params = struct();
ga_params.populationSize = 100;        % Increased for more diversity
ga_params.maxGenerations = 100;
ga_params.crossoverRate = 0.8;
ga_params.mutationRate = 0.05;         % Slightly increased for more exploration
ga_params.eliteCount = 3;
ga_params.tournamentSize = 5;

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

% Evaluate on validation set
val_pred = predict(Theta1, Theta2, X_val);
val_accuracy = mean(double(val_pred == y_val)) * 100;
fprintf('Validation set accuracy: %.2f%%\n', val_accuracy);

% Evaluate on writer-dependent set
wd_pred = predict(Theta1, Theta2, X_wd);
wd_accuracy = mean(double(wd_pred == y_wd)) * 100;
fprintf('Writer-dependent accuracy: %.2f%%\n', wd_accuracy);

% Evaluate on writer-independent set
wi_pred = predict(Theta1, Theta2, X_wi);
wi_accuracy = mean(double(wi_pred == y_wi)) * 100;
fprintf('Writer-independent accuracy: %.2f%%\n', wi_accuracy);

% Create confusion matrices
plotConfusionMatrix(y_val, val_pred, 'Validation');
plotConfusionMatrix(y_wd, wd_pred, 'Writer-Dependent');
plotConfusionMatrix(y_wi, wi_pred, 'Writer-Independent');
