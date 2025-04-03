function [input, hidden, output, initt1, initt2] = ANN(A_train, B_train)
  input = size(A_train, 2);
  hidden = 1;
  output = length(unique(B_train));

  initt1 = randInitializeWeights(input, hidden);
  initt2 = randInitializeWeights(hidden, output);

end
