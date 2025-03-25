function normalized = normalization(data)
  normalized = (data - min(data)) ./ (max(data) - min(data));
end
