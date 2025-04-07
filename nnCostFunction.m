function [J] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

J = 0;

%without regularization
y_temp = eye(num_labels);

%input layer to hidden layer
X = [ones(size(X, 1), 1) X]; %adds column of ones
z2 = X*Theta1';
a2 = sigmoid(z2);

%hidden layer to output layer
a2 = [ones(size(a2, 1), 1) a2]; %adds column of ones
z3 = a2*Theta2';
h = sigmoid(z3);


Y = zeros(m, num_labels);
for i = 1 : m
  Y(i,:) = y_temp(y(i), :);
end

J = sum(sum((-Y) .* log(h) - (1 - Y) .* log(1 - h)))/m;


%with regularization
%cost function
J = J + ((lambda/(2*m))*(sum(sum(Theta1(:, [2:columns(Theta1)]).*Theta1(:, [2:columns(Theta1)]))) + sum(sum(Theta2(:, [2:columns(Theta2)]).*Theta2(:, [2:columns(Theta2)])))));

end
