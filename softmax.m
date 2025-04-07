function S = softmax(X)
  e_X = exp(X - max(X, [], 2));
  S = e_X ./ sum(e_X, 2);
end
