function [accuracy, correct, predictions] = accuracy(best_weights, input_size, hidden_size, output_size, X_test, Y_test)
  theta1_elems = hidden_size * (input_size + 1);
  theta2_elems = output_size * (hidden_size + 1);

  Theta1 = reshape(best_weights(1:theta1_elems), hidden_size, input_size + 1);
  Theta2 = reshape(best_weights(theta1_elems+1:end), output_size, hidden_size + 1);

  predictions = predict(Theta1, Theta2, X_test);

  correct = sum(predictions == Y_test);
  accuracy = (correct / length(Y_test)) * 100;
end


