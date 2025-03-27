function [data, labels] = loadData(datafile)
  file = fopen(datafile, 'r');

  inst = 24;
  att = 4;
  data = zeros(inst, att);
  labels = zeros(inst, 1);

  for i = 1:inst
    line = fgetl(file);

    values = strsplit(line);

    data(i, :) = str2double(values(2:5));
    labels(i) = str2double(values(6));
  end

  fclose(file);

  normalized = normalization(data);
  [A_train, B_train, X_test, Y_test] = test_train(normalized, labels, 0.6);
  [input, hidden, unq, initt1, initt2] = ANN(A_train, B_train);
  [population, pop, crossrate, muterate] = GA(initt1, initt2, A_train, B_train);

  disp('Data:');
  disp(data);
  disp('Labels:');
  disp(labels);
  disp('Normalized:');
  disp(normalized);
  disp('Training set:');
  disp(A_train);
  disp('Training labels:');
  disp(B_train);
  disp('Testing set:');
  disp(X_test);
  disp('Testing labels:');
  disp(Y_test);
  disp('Theta1:');
  disp(initt1);
  disp('Theta2:');
  disp(initt2);
  disp('Population:');
  disp(population);

  popu = population;

  for i=1:100
    [new_pop, best_fit] = evolution(popu, A_train, B_train, input, hidden, unq, 1, pop, crossrate, muterate);
    popu = new_pop;
    disp(['Generation ', num2str(i), ' | Best Fitness: ', num2str(best_fit)]);
  endfor

  fitness = zeros(pop, 1);
  for i = 1:pop
      fitness(i) = nnCostFunction(popu(i,:), input, hidden, unq, A_train, B_train, 1);
  end
  [best_fit, best_idx] = min(fitness);
  best_params = popu(best_idx, :);

  % Reshape into Theta1 and Theta2 using the best solutions
  Theta1 = reshape(best_params(1:hidden * (input + 1)), hidden, (input + 1));
  Theta2 = reshape(best_params((hidden * (input + 1)) + 1:end), unq, (hidden + 1));

  % for prediction and getting accuracy
  train_pred = predict(Theta1, Theta2, A_train);
  test_pred = predict(Theta1, Theta2, X_test);

  train_accuracy = mean(double(train_pred == B_train)) * 100;
  test_accuracy = mean(double(test_pred == Y_test)) * 100;

  disp('=== Final Results ===');
  disp(['Training Accuracy: ', num2str(train_accuracy), '%']);
  disp(['Testing Accuracy: ', num2str(test_accuracy), '%']);
end
