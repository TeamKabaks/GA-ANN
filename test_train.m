function [A_train, B_train, X_test, Y_test] = test_train(samples, images, labels, train_ratio)
  % Ensure inputs are correct size
    if size(images, 1) ~= samples || size(labels, 1) ~= samples
        error('Mismatch between sample count and data size');
    end

    % Shuffle data before splitting
    rng(42); % For reproducibility
    shuffle_idx = randperm(samples);
    images = images(shuffle_idx, :);
    labels = labels(shuffle_idx);

    % Calculate split sizes
    num_train = floor(train_ratio * samples);

    % Split features
    A_train = images(1:num_train, :);
    X_test = images(num_train+1:end, :);

    % Split labels
    B_train = labels(1:num_train);
    Y_test = labels(num_train+1:end);

    % Verify class distribution (optional debug output)
    train_classes = unique(B_train);
    test_classes = unique(Y_test);
    disp(['Training set classes: ', num2str(train_classes')]);
    disp(['Test set classes: ', num2str(test_classes')]);

    % Check for missing classes in either set
    if length(train_classes) ~= length(test_classes)
        warning('Class distribution mismatch between train and test sets');
    end
end
