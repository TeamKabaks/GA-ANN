function loadData()
  output_DIR = 'C:\Users\IDEAPAD\Documents\GitHub\GA-ANN\Graphs';
  if ~exist(output_DIR, 'dir')
    mkdir(output_DIR);
  end

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
  gens = 5;

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

  % plotting the best costs (fitness)
  plotFitness(best_costs, output_DIR);

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

  % added [acc, predictions] for confusion matrix
  [acc, predictions] = accuracy(popu(1, :), input, hidden, unq, X_test, Y_test);

  plotConfusionMatrix(Y_test, predictions, unq, output_DIR);
  title(sprintf('Confusion Matrix (Accuracy: %.2f%%)', acc));

end

function plotFitness(best_costs, output)
  fitness_fig = figure('Visible', 'off');
  plot(1:length(best_costs), best_costs, 'b-', 'LineWidth', 2);
  title('Fitness over Generations');
  xlabel('Generation');
  ylabel('Best Fitness');
  grid on;
  set(gca, 'FontSize', 12);

  saveas(fitness_fig, fullfile(output, 'fitness_graph_second.png'));
  saveas(fitness_fig, fullfile(output, 'fitness_graph_second.fig'));
  close(fitness_fig);

end

function plotConfusionMatrix(true_labels, predicted_labels, num_classes, output)
  cm = zeros(num_classes, num_classes);
  for i = 1:length(true_labels)
    cm(true_labels(i), predicted_labels(i)) = cm(true_labels(i), predicted_labels(i)) + 1;
  end

  cm_normalized = cm ./ sum(cm, 2);

  cm_fig = figure('Visible', 'off');
  imagesc(cm_normalized);
  colormap(flipud(gray));
  colorbar;
  title('Normalized Confusion Matrix');
  xlabel('Predicted Label');
  ylabel('True Label');

  % Set the ticks and labels
  tick_positions = 1:num_classes;
  tick_labels = arrayfun(@num2str, 0:num_classes-1, 'UniformOutput', false);
  xticks(tick_positions);
  yticks(tick_positions);
  xticklabels(tick_labels);
  yticklabels(tick_labels);

  for i = 1:num_classes
    for j = 1:num_classes
        if cm_normalized(i, j) > 0.5
            text_color = 'w';
        else
            text_color = 'k';
        end
        text(j, i, sprintf('%.2f', cm_normalized(i, j)), ...
             'HorizontalAlignment', 'center', ...
             'Color', text_color);
    end
  end

  saveas(cm_fig, fullfile(output, 'confusion_matrix_second.png'));
  saveas(cm_fig, fullfile(output, 'confusion_matrix_second.fig'));
  close(cm_fig);
end
