function [population, pop, crossrate, muterate, lambda] = GA(initt1, initt2, A_train, B_train, samples)
  pop = min(200, samples);
  crossrate = 0.8;
  muterate = 0.2;

  lambda = 0.1;

  population = zeros(pop, numel(initt1) + numel(initt2));

  for i = 1:pop

    input_size = size(initt1, 2) - 1; % Subtract bias
    hidden_size = size(initt1, 1);
    output_size = size(initt2, 1);

    scale1 = sqrt(6/(input_size + hidden_size));
    scale2 = sqrt(6/(hidden_size + output_size));

    population(i,:) = [rand(size(initt1(:)))', rand(size(initt2(:)))'];
  end

  fprintf('Genetic Algorithm Parameters:\n');
  fprintf('Population size: %d\n', pop);
  fprintf('Crossover rate: %.2f\n', crossrate);
  fprintf('Mutation rate: %.2f\n', muterate);
end
