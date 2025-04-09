function [population, pop, crossrate, muterate] = GA(initt1, initt2, A_train, B_train)
  pop = 200;
  crossrate = 0.75;
  muterate = 0.05;

  weight = [initt1(:); initt2(:)];

  population = zeros(pop, length(weight));
  for i = 1:pop
    population(i, :) = weight + randn(size(weight)) * 0.1;
  end
end
