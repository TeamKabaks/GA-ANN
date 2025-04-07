function [A_normalized, X_normalized] = normalization(train_data, test_data)
  A_normalized = (train_data - min(train_data)) ./ (max(train_data) - min(train_data));
  X_normalized = (test_data - min(test_data)) ./ (max(test_data) - min(test_data));
end
