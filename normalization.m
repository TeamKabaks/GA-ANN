function normalized = normalization(matrices)
  samples = 1934;

  % Normalize each image separately to [0,1]
  for i = 1:samples
    matrices(i,:) = (matrices(i,:) - min(matrices(i,:))) / (max(matrices(i,:)) - min(matrices(i,:)) + eps);
  end
end
