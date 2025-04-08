function loadData()
  totalics = tic;
  tra_data = dlmread('optdigits.tra', ',');
  tes_data = dlmread('optdigits.tes', ',');

  tic;
  A_train = tra_data(:, 1:64) / 16.0;
  B_train = tra_data(:, 65) + 1;
  X_test = tes_data(:, 1:64) / 16.0;
  Y_test = tes_data(:, 65) + 1;
  data_load_time = toc;
  fprintf('\nData loading time: %.10fs\n', data_load_time);

  tic;
  [input, hidden, unq, initt1, initt2] = ANN(A_train, B_train);
  [population, pop, crossrate, muterate] = GA(initt1, initt2, A_train, B_train);
  init_time = toc;
  fprintf('Initialization time: %.10fs\n', init_time);

  %disp('Normalized Training Data:');
  %disp(A_train(5, :));
  %disp('Labels:');
  %disp(B_train(5, :));
  %disp('Normalized Testing Set:');
  %disp(X_test(5, :));
  %disp('Testing labels:');
  %disp(Y_test(5, :));

  popu = population;
  gens = 100;

  time_per_generation = zeros(gens, 1);
  best_costs = zeros(gens, 1);
  for i=1:gens
    gen_tic = tic;
    [new_pop, best_fit] = evolution(popu, A_train, B_train, input, hidden, unq, 1, pop, crossrate, muterate);
    popu = new_pop;
    best_costs(i) = best_fit;
    time_per_generation(i) = toc(gen_tic);
    fprintf('\nGeneration %d | Best Fitness: %.5f | Runtime: %.10fs', i, best_fit, time_per_generation(i));
  endfor

  total_time = toc(totalics);
  fprintf('\n\nTotal Runtime: %.10fs\n', total_time);
  fprintf('\n=== Parameters Used ===\n');
  fprintf('Population Count: %d\n', pop);
  fprintf('Number of Generations: %d\n', gens);
  fprintf('Crossover Rate: %.2f%%\n', crossrate * 100);
  fprintf('Mutation Rate: %.2f%%\n', muterate * 100);
  fprintf('Number of Input Neurons: %d\n', input);
  fprintf('Number of Hidden Neurons: %d\n', hidden);
  fprintf('Number of Output Neurons: %d\n', unq);
  fprintf('Average Time per Generation: %.10fs\n', mean(time_per_generation));
  accuracy(popu(1, :), input, hidden, unq, X_test, Y_test);
end
