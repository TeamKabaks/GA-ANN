function [population, pop_size, crossrate, muterate, lambda] = GA(initt1, initt2, initt3, train_data, train_labels, samples)
  pop_size = 100;
  crossrate = 0.75;
  muterate = 0.3;

  lambda = 1;

  weight = [initt1(:); initt2(:)];

  population = zeros(pop_size, length(weight));
  for i = 1:pop_size
    population(i, :) = weight + randn(size(weight)) * 0.1;
  end

  fprintf('Genetic Algorithm Parameters:\n');
  fprintf('Population size: %d\n', pop_size);
  fprintf('Crossover rate: %.2f\n', crossrate);
  fprintf('Mutation rate: %.2f\n', muterate);
end
