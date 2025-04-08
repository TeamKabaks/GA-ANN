function [new_pop, best_fit] = evolution(population, A_train, B_train, input, hidden, output, lambda, pop, crossrate, muterate)
    fitness = zeros(pop, 1);
    for i = 1:pop
        fitness(i) = nnCostFunction(population(i,:), input, hidden, output, A_train, B_train, lambda);
    end

    parents = zeros(pop, size(population, 2));
    for i = 1:2:pop
        candidates = randperm(pop, 5);
        [~, sorted_idx] = sort(fitness(candidates));
        parent1 = population(candidates(sorted_idx(1)), :);
        parent2 = population(candidates(sorted_idx(2)), :);
        parents(i, :) = parent1;
        if i+1 <= pop
            parents(i+1, :) = parent2;
        end
    end

    offspring = parents;
    for i = 1:2:pop-1
        if rand() < crossrate
            crosspoint = randi([1, size(population, 2)-1]);
            temp = offspring(i, crosspoint+1:end);
            offspring(i, crosspoint+1:end) = offspring(i+1, crosspoint+1:end);
            offspring(i+1, crosspoint+1:end) = temp;
        end
    end

    for i = 1:pop
        mutemask = rand(1, size(offspring,2)) < muterate;
        offspring(i, mutemask) = offspring(i, mutemask) + 0.1*randn(1, sum(mutemask));
    end

    [sorted_fit, sorted_idx] = sort(fitness);
    elite_count = max(2, floor(0.05 * pop));
    offspring(1:elite_count,:) = population(sorted_idx(1:elite_count), :);

    new_pop = offspring;
    best_fit = sorted_fit(1);
end
