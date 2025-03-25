function [new_pop, best_fit] = evolution(population, A_train, B_train, input, hidden, unq, lambda, pop, crossrate, muterate, X_test, Y_test, Theta1, Theta2)
  fitness = zeros(pop, 1);
  for i = 1:pop
    fitness(i) = nnCostFunction([Theta1(:); Theta2(:)], input, hidden, unq, A_train, B_train, lambda);
  end

  fit = sort(fitness);

  parents = zeros(pop, size(population, 2));
  for i = 1:pop
    candidates = randperm(pop, 2);
    fitval = fitness(candidates);
    [minimum, index] = min(fitval);
    fit_parent = population(candidates(index), :);
    parents(i, :) = fit_parent;
  end

  offspring = parents;
  for i = 1:2:pop-1
    if rand() < crossrate
      crosspoint = randi([1, size(population, 2) - 1]);
      offspring(i, crosspoint+1:end) = parents(i+1, crosspoint+1:end);
      offspring(i+1, crosspoint+1:end) = parents(i, crosspoint+1:end);
    end
  end

  for i = 1:pop
    if rand() < muterate
      mutemask = rand(size(offspring(i, :))) < 0.1;
      offspring(i, mutemask) = offspring(i, mutemask) + 0.05 * randn(1, sum(mutemask));
    end
  end

  [best_fit, index_best] = min(fitness);
  offspring(1, :) = population(index_best, :);

  new_pop = offspring;
end
