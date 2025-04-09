function [input, hidden, unq, initt1, initt2] = ANN(A_train, B_train)
  input = size(A_train, 2);
  hidden = 96;
  unq = length(unique(B_train));

  initt1 = randInitializeWeights(input, hidden);
  initt2 = randInitializeWeights(hidden, unq);
end
