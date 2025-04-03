function [data, labels] = loadData(datafile)
  raw_data = dlmread(datafile);

  samples = floor(size(raw_data, 1) / 33);
  fprintf('Processing %d samples...\n', samples);

  matrices = zeros(samples, 32*32);
  labels = zeros(samples, 1);

  for i = 1:samples
    start_idx = (i - 1) * 33 + 1;
    digit = raw_data(start_idx:start_idx+31, :);
    matrices(i, :) = digit(:)'; % for flattening the matrix
    labels(i) = raw_data(start_idx+32, 1);
  end

  min_val = min(matrices(:));
  max_val = max(matrices(:));
  matrices = (matrices - min_val) / (max_val - min_val + eps);

  for i = 1:samples
    curr_min = min(matrices(i,:));
    curr_max = max(matrices(i,:));
    if (curr_max - curr_min) > 0.1 % Only normalize if there's sufficient contrast
      matrices(i,:) = (matrices(i,:) - curr_min) / (curr_max - curr_min + eps);
    end
  end

  [A_train, B_train, X_test, Y_test] = test_train(samples, matrices, labels, 0.7);
  [input, hidden, output, initt1, initt2] = ANN(A_train, B_train);
  [population, pop, crossrate, muterate, lambda] = GA(initt1, initt2, A_train, B_train, samples);

  popu = population;

  best_fitness_history = zeros(100,1);


  %validation set for early stopping
  val_size = floor(0.2 * size(A_train, 1));
  A_val = A_train(1:val_size, :);
  B_val = B_train(1:val_size);
  A_train_reduced = A_train(val_size+1:end, :);
  B_train_reduced = B_train(val_size+1:end);


  for i=1:100
    [new_pop, best_fit] = evolution(popu, A_train, B_train, input, hidden, output, lambda, pop, crossrate, muterate);
    best_fitness_history(i) = best_fit;
    popu = new_pop;
    % Early stopping if fitness plateaus
    if i > 20 && std(best_fitness_history(i-19:i)) < 1e-4
      break;
    end
    disp(['Generation ', num2str(i), ' | Best Fitness: ', num2str(best_fit)]);
    endfor

  fitness = zeros(pop, 1);
  for i = 1:pop
      fitness(i) = nnCostFunction(popu(i,:), input, hidden, output, A_train_reduced, B_train_reduced, lambda);
  end
  avg_fitness_history(i) = mean(fitness);

  % getting the best solution
  [best_fit, best_idx] = min(fitness);
  best_params = popu(best_idx, :);

  % Reshape into Theta1 and Theta2 using the best solutions
  Theta1 = reshape(best_params(1:hidden * (input + 1)), hidden, (input + 1));
  Theta2 = reshape(best_params((hidden * (input + 1)) + 1:end), output, (hidden + 1));

  % for prediction and getting accuracy
  val_pred = predict(Theta1, Theta2, A_val);
  train_pred = predict(Theta1, Theta2, A_train);
  test_pred = predict(Theta1, Theta2, X_test);

  [train_accuracy, test_accuracy] = calculate_accuracy(train_pred, B_train, test_pred, Y_test);

  % Plot training progress
  figure;
  plot(1:i, best_fitness_history(1:i), 'b-', 'LineWidth', 2);
  hold on;
  plot(1:i, avg_fitness_history(1:i), 'r--', 'LineWidth', 1.5);
  xlabel('Generation');
  ylabel('Cost Function');
  title('Training Progress');
  legend('Best Fitness', 'Average Fitness');
  grid on;

  disp('=== Final Results ===');
  disp(['Training Accuracy: ', num2str(train_accuracy), '%']);
  disp(['Testing Accuracy: ', num2str(test_accuracy), '%']);

  data = matrices;
end
