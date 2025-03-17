function [best_nn_params, best_J] = ga_optimize_nn(input_layer_size, hidden_layer_size, num_labels, X, y, lambda, ga_params)
% GA_OPTIMIZE_NN Uses genetic algorithm to optimize neural network weights
%   Inputs:
%     input_layer_size  - Size of the input layer
%     hidden_layer_size - Size of the hidden layer
%     num_labels        - Number of output classes/labels
%     X                 - Training data features
%     y                 - Training data labels
%     lambda            - Regularization parameter
%     ga_params         - Structure with GA parameters
%   Outputs:
%     best_nn_params    - Best weights found
%     best_J            - Best cost achieved

% Default GA parameters if not provided
if nargin < 7
    ga_params = struct();
end
if ~isfield(ga_params, 'populationSize'), ga_params.populationSize = 50; end
if ~isfield(ga_params, 'maxGenerations'), ga_params.maxGenerations = 100; end
if ~isfield(ga_params, 'crossoverRate'), ga_params.crossoverRate = 0.8; end
if ~isfield(ga_params, 'mutationRate'), ga_params.mutationRate = 0.03; end
if ~isfield(ga_params, 'eliteCount'), ga_params.eliteCount = 2; end
if ~isfield(ga_params, 'tournamentSize'), ga_params.tournamentSize = 3; end

% Calculate total number of weights
total_weights = hidden_layer_size * (input_layer_size + 1) + ...
                num_labels * (hidden_layer_size + 1);

% Initialize population
population = zeros(ga_params.populationSize, total_weights);
for i = 1:ga_params.populationSize
    % Initialize weights for each chromosome
    Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
    Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

    % Flatten the weights into a single vector (chromosome)
    population(i,:) = [Theta1(:)' Theta2(:)'];
end

% Initialize tracking
fitness_history = zeros(ga_params.maxGenerations, 1);
best_fitness_history = zeros(ga_params.maxGenerations, 1);
mean_fitness_history = zeros(ga_params.maxGenerations, 1);
best_J = Inf;
best_nn_params = [];

% Main GA loop
for generation = 1:ga_params.maxGenerations
    % 1. Evaluate fitness (lower cost = higher fitness)
    costs = zeros(ga_params.populationSize, 1);
    for i = 1:ga_params.populationSize
        costs(i) = nnCostFunction(population(i,:), input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
    end

    % Convert costs to fitness (higher is better)
    max_cost = max(costs) + 1; % Add 1 to ensure all positive
    fitness = max_cost - costs; % Invert so lower cost = higher fitness

    % Update best solution found
    [min_cost, min_idx] = min(costs);
    if min_cost < best_J
        best_J = min_cost;
        best_nn_params = population(min_idx, :);
    end

    % Update history
    [best_fit, ~] = max(fitness);
    best_fitness_history(generation) = best_fit;
    mean_fitness_history(generation) = mean(fitness);

    % Display progress
    if mod(generation, 10) == 0 || generation == 1
        fprintf('Generation %d: Best Cost = %.6f, Mean Cost = %.6f\n', ...
                generation, min_cost, mean(costs));
    end

    % 2. Selection through tournament selection
    new_population = zeros(size(population));

    % Elitism - copy best individuals directly
    [~, elite_indices] = sort(costs);
    elite_indices = elite_indices(1:ga_params.eliteCount);
    new_population(1:ga_params.eliteCount, :) = population(elite_indices, :);

    % Tournament selection for the rest
    for i = ga_params.eliteCount+1:ga_params.populationSize
        % Select tournament participants
        tournament_idx = randi(ga_params.populationSize, 1, ga_params.tournamentSize);
        [~, winner_pos] = max(fitness(tournament_idx));
        winner_idx = tournament_idx(winner_pos);
        new_population(i, :) = population(winner_idx, :);
    end

    % 3. Crossover
    for i = ga_params.eliteCount+1:2:ga_params.populationSize-1
        if rand < ga_params.crossoverRate
            % Select crossover point
            point = randi(total_weights - 1);

            % Store original individuals
            parent1 = new_population(i, :);
            parent2 = new_population(i+1, :);

            % Perform crossover
            new_population(i, :) = [parent1(1:point), parent2(point+1:end)];
            new_population(i+1, :) = [parent2(1:point), parent1(point+1:end)];
        end
    end

    % 4. Mutation
    for i = ga_params.eliteCount+1:ga_params.populationSize
        for j = 1:total_weights
            if rand < ga_params.mutationRate
                % Add small Gaussian noise
                new_population(i, j) = new_population(i, j) + randn * 0.1;
            end
        end
    end

    % 5. Replace population
    population = new_population;
end

% Plot fitness history
figure;
plot(1:ga_params.maxGenerations, best_fitness_history, 'b-', ...
     1:ga_params.maxGenerations, mean_fitness_history, 'r-');
legend('Best Fitness', 'Mean Fitness');
title('Fitness History');
xlabel('Generation');
ylabel('Fitness');

end

% Helper function to evaluate accuracy
function accuracy = evaluateAccuracy(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y)
    % Reshape nn_params back into weight matrices
    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                    hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                    num_labels, (hidden_layer_size + 1));

    % Make predictions
    pred = predict(Theta1, Theta2, X);

    % Calculate accuracy
    accuracy = mean(double(pred == y)) * 100;
end
