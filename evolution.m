function [new_pop, best_fit] = evolution(population, A_train, B_train, input, hidden, output, lambda, pop, crossrate, muterate)
    fitness = zeros(pop, 1);
    for i = 1:pop
        fitness(i) = nnCostFunction(population(i,:), input, hidden, output, A_train, B_train, lambda);
    end

    [sorted_fit, sorted_idx] = sort(fitness);
    remaining = sorted_idx;

    parents = zeros(pop, size(population, 2));
    for i = 1:pop
        if length(remaining) < 2
            [~, remaining] = sort(fitness);  % Reset if pool exhausted
        end

        tournament_size = min(3, length(remaining));
        candidates = remaining(1:tournament_size);

        pair = randperm(tournament_size, 2);
        index1 = candidates(pair(1));
        index2 = candidates(pair(2));

        if fitness(index1) < fitness(index2)
            selected = index1;
        else
            selected = index2;
        end

        parents(i,:) = population(selected,:);
        remaining(remaining == selected) = [];
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
        offspring(i, mutemask) = offspring(i, mutemask) + 0.05*randn(1, sum(mutemask));
    end

    elite_count = max(1, floor(0.1*pop));  % Ensure at least 1 elite
    offspring(1:elite_count,:) = population(sorted_idx(1:elite_count),:);

    new_pop = offspring;
    best_fit = sorted_fit(1);
end
