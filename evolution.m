function [new_pop, best_fit] = evolution(population, A_train, B_train, input, hidden, output, lambda, pop, crossrate, muterate)
  % Fitness evaluation
  fitness = zeros(pop, 1);
  for i = 1:pop
    fitness(i) = nnCostFunction(population(i,:), input, hidden, output, A_train, B_train, lambda);
  end

  % Sort fitness
  [sorted_fit, sorted_idx] = sort(fitness);

  parents = zeros(pop, size(population, 2));
  tournament_size = 5;
  for i = 1:pop
    candidates = randperm(pop, tournament_size);
    candidate_fitness = fitness(candidates);
    [~, best_idx] = min(candidate_fitness);
    parents(i, :) = population(candidates(best_idx), :);
    %parents(i, :) = fit_parent;
  end

  % Crossover
  offspring = parents;
  for i = 1:2:pop-1
    if rand() < crossrate
      crosspoint = randi([1, size(population, 2) - 1]);
      offspring(i, crosspoint+1:end) = parents(i+1, crosspoint+1:end);
      offspring(i+1, crosspoint+1:end) = parents(i, crosspoint+1:end);
    end
  end

  % Mutation with adaptive scaling
  [~, fitness_rank] = sort(fitness);
  for i = 1:pop
    if rand() < muterate
      % Scale mutation rate based on fitness rank
      rank = find(fitness_rank == i);
      scale = 0.2 * (1 - (rank/pop)^2);  % Better solutions get smaller mutations

      % only mutate subset of genes
      num_params = size(offspring(i,:), 2);
      mutation_points = rand(1, num_params) < 0.1; % mutate ~10% of genes

      mutation = zeros(1, num_params);
      mutation(mutation_points) = scale * randn(1, sum(mutation_points));
      %mutation = scale * randn(size(offspring(i,:)));
      offspring(i,:) = offspring(i,:) + mutation;
      end
  end

  % Elitism - keep top 10% of solutions
  elite_count = floor(0.1 * pop);
  for i = 1:elite_count
    offspring(i, :) = population(sorted_idx(i), :);
  end

  % Added diversity preservation mechanism
  if i > 20 && std(sorted_fit(1:10)) < 1e-5
    % If top solutions are too similar, introduce diversity
    for i = floor(0.8*pop):pop
      offspring(i,:) = offspring(i,:) + 0.5 * randn(size(offspring(i,:)));
    end
  end

  new_pop = offspring;
  best_fit = sorted_fit(1);
end
