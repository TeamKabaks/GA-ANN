function [input, hidden, output, initt1, initt2] = ANN(train_data, train_labels)
  input = size(train_data, 2);
  hidden = 1;
  output = length(unique(train_labels));

  initt1 = randInitializeWeights(input, hidden);
  initt2 = randInitializeWeights(hidden, output);
end
