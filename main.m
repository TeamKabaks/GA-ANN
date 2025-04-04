function main(train_file, wdep_test_file, windep_test_file)
  [train_data, train_labels, val_data, val_labels, wdep_test_data, wdep_test_labels, windep_test_data, windep_test_labels] = loadData(train_file, wdep_test_file, windep_test_file)

  fprintf('Data loaded successfully:\n');
  fprintf('  - Training set: %d samples\n', size(train_data, 1));
  fprintf('  - Validation set: %d samples\n', size(val_data, 1));
  fprintf('  - Writer-dependent test set: %d samples\n', size(wdep_test_data, 1));
  fprintf('  - Writer-independent test set: %d samples\n', size(windep_test_data, 1));

  samples = size(train_data, 1);

  [input, hidden, output, initt1, initt2] = ANN(train_data, train_labels);
  [population, pop_size, crossrate, muterate, lambda] = GA(initt1, initt2, train_data, train_labels, samples);

  popu = population;

  best_fitness_history = zeros(200,1);

  for i=1:200
    [new_pop, best_fit] = evolution(popu, train_data, train_labels, input, hidden, output, lambda, pop_size, crossrate, muterate);
    best_fitness_history(i) = best_fit;
    popu = new_pop;
    disp(['Generation ', num2str(i), ' | Best Fitness: ', num2str(best_fit)]);
    endfor

  fitness = zeros(pop_size, 1);
  for i = 1:pop_size
      fitness(i) = nnCostFunction(popu(i,:), input, hidden, output, train_data, train_labels, lambda);
  end
  %avg_fitness_history(i) = mean(fitness);

  % getting the best solution
  [~, best_idx] = min(fitness);
  best_params = popu(best_idx, :);

  % Reshape into Theta1 and Theta2 using the best solutions
  Theta1 = reshape(best_params(1:hidden * (input + 1)), hidden, (input + 1));
  Theta2 = reshape(best_params((hidden * (input + 1)) + 1:end), output, (hidden + 1));

  %% Evaluate Performance on All Sets

  % Training set
  train_pred = predict(Theta1, Theta2, train_data);
  train_accuracy = mean(double(train_pred == train_labels)) * 100;

  % Validation set
  val_pred = predict(Theta1, Theta2, val_data);
  val_accuracy = mean(double(val_pred == val_labels)) * 100;

  % Writer-dependent test set
  wdep_pred = predict(Theta1, Theta2, wdep_test_data);
  wdep_accuracy = mean(double(wdep_pred == wdep_test_labels)) * 100;

  % Writer-independent test set
  windep_pred = predict(Theta1, Theta2, windep_test_data);
  windep_accuracy = mean(double(windep_pred == windep_test_labels)) * 100;

  %% Display Results
  fprintf('\n=== Final Results ===\n');
  fprintf('Training Accuracy: %.2f%%\n', train_accuracy);
  fprintf('Validation Accuracy: %.2f%%\n', val_accuracy);
  fprintf('Writer-Dependent Test Accuracy: %.2f%%\n', wdep_accuracy);
  fprintf('Writer-Independent Test Accuracy: %.2f%%\n', windep_accuracy);

  %% Console Confusion Matrices
  fprintf('\n=== Confusion Matrices ===\n');

  % Training set confusion matrix
  fprintf('\nTraining Set Confusion Matrix (Accuracy: %.1f%%):\n', train_accuracy);
  printConfusionMatrix(train_labels, train_pred);

  % Validation set confusion matrix
  fprintf('\nValidation Set Confusion Matrix (Accuracy: %.1f%%):\n', val_accuracy);
  printConfusionMatrix(val_labels, val_pred);

  % Writer-dependent test set confusion matrix
  fprintf('\nWriter-Dependent Test Confusion Matrix (Accuracy: %.1f%%):\n', wdep_accuracy);
  printConfusionMatrix(wdep_test_labels, wdep_pred);

  % Writer-independent test set confusion matrix
  fprintf('\nWriter-Independent Test Confusion Matrix (Accuracy: %.1f%%):\n', windep_accuracy);
  printConfusionMatrix(windep_test_labels, windep_pred);

  figure;
  plot(1:100, best_fitness_history, 'b-', 'LineWidth', 2);
  xlabel('Generation');
  ylabel('Cost Function');
  title('Training Progress');
  legend('Best Fitness', 'Average Fitness');
  grid on;
end

function printConfusionMatrix(true_labels, pred_labels)
  % Get unique classes
  classes = unique([true_labels; pred_labels]);
  num_classes = length(classes);

  % Initialize confusion matrix
  conf_mat = zeros(num_classes, num_classes);

  % Populate confusion matrix
  for i = 1:length(true_labels)
    true_idx = find(classes == true_labels(i));
    pred_idx = find(classes == pred_labels(i));
    conf_mat(true_idx, pred_idx) = conf_mat(true_idx, pred_idx) + 1;
  end

  % Print header row
  fprintf('\nActual \\ Predicted');
  for c = 1:num_classes
    fprintf('%6d', classes(c));
  end
  fprintf('\n');

  % Print each row of the matrix
  for true_c = 1:num_classes
    fprintf('%6d', classes(true_c));
    for pred_c = 1:num_classes
      fprintf('%6d', conf_mat(true_c, pred_c));
    end
    fprintf('\n');
  end

  % Print class-wise accuracy
  fprintf('\nClass-wise accuracy:\n');
  for c = 1:num_classes
    accuracy = conf_mat(c,c) / sum(conf_mat(c,:)) * 100;
    fprintf('Class %d: %.1f%%\n', classes(c), accuracy);
  end
end
