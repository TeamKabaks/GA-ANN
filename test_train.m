function [A_train, B_train, X_test, Y_test] = test_train(normalized, labels, train_ratio)
  n = 24;
  num_train = floor(train_ratio * n);

  A_train = normalized(1:num_train, :);
  B_train = labels(1:num_train);
  X_test = normalized(num_train+1:end, :);
  Y_test = labels(num_train+1:end);
end
