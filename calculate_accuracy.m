function [train_accuracy, test_accuracy] = calculate_accuracy(train_pred, B_train, test_pred, Y_test)
    % VALIDATE LABELS FIRST
    % Check for valid label ranges and types
    validate_labels(B_train);
    validate_labels(Y_test);

    % Ensure predictions match label format
    train_pred = align_labels(train_pred, B_train);
    test_pred = align_labels(test_pred, Y_test);

    % CALCULATE ACCURACY WITH DETAILED VALIDATION
    train_accuracy = compute_accuracy_with_validation(train_pred, B_train, 'Training');
    test_accuracy = compute_accuracy_with_validation(test_pred, Y_test, 'Testing');

    % DISPLAY COMPREHENSIVE RESULTS
    display_results(train_pred, B_train, test_pred, Y_test, train_accuracy, test_accuracy);
end

%% Helper Functions

function validate_labels(labels)
    % Check for empty labels
    if isempty(labels)
        error('Label array is empty');
    end

    % Check label type (should be numeric)
    if ~isnumeric(labels)
        error('Labels must be numeric');
    end

    % Check for NaN or Inf values
    if any(isnan(labels)) || any(isinf(labels))
        error('Labels contain NaN or Inf values');
    end

    % Check if labels are integers
    if any(rem(labels, 1) ~= 0)
        warning('Some labels are not integers - are you sure this is correct?');
    end

    % Verify reasonable range (adjust for your dataset)
    if min(labels) < 0 || max(labels) > 9
        warning('Labels outside expected 0-9 range - verify label encoding');
    end
end

function aligned_pred = align_labels(predictions, true_labels)
    % Handle 0-based vs 1-based label mismatch
    pred_min = min(predictions);
    true_min = min(true_labels);

    if pred_min ~= true_min
        if pred_min == 1 && true_min == 0
            aligned_pred = predictions - 1;
            disp('Adjusted predictions: Converted from 1-based to 0-based');
        elseif pred_min == 0 && true_min == 1
            aligned_pred = predictions + 1;
            disp('Adjusted predictions: Converted from 0-based to 1-based');
        else
            warning('Label base mismatch - predictions may be incorrect');
            aligned_pred = predictions;
        end
    else
        aligned_pred = predictions;
    end
end

function accuracy = compute_accuracy_with_validation(predictions, true_labels, set_name)
    % Dimension check
    if ~isequal(size(predictions), size(true_labels))
        error('%s set predictions/labels size mismatch', set_name);
    end

    % Calculate accuracy - simpler comparison for integer labels
    correct = predictions == true_labels;
    accuracy = mean(correct) * 100;

    % Additional validation with improved threshold
    if accuracy < 20 && strcmp(set_name, 'Testing')
        warning('Low %s accuracy (%.2f%%) - check for potential issues', set_name, accuracy);
    end
end

function display_results(train_pred, B_train, test_pred, Y_test, train_acc, test_acc)
    % Basic accuracy display
    fprintf('\n=== Accuracy Results ===\n');
    fprintf('Training Accuracy: %.2f%%\n', train_acc);
    fprintf('Testing Accuracy:  %.2f%%\n\n', test_acc);

    % Confusion matrices - with fallback if Statistics toolbox isn't available
    fprintf('Training Confusion Matrix:\n');
    try
        print_confusion_matrix(B_train, train_pred);
    catch
        fprintf('Could not generate confusion matrix (requires Statistics Toolbox)\n');
        % Simple alternative
        print_simple_metrics(B_train, train_pred);
    end

    fprintf('\nTesting Confusion Matrix:\n');
    try
        print_confusion_matrix(Y_test, test_pred);
    catch
        fprintf('Could not generate confusion matrix (requires Statistics Toolbox)\n');
        % Simple alternative
        print_simple_metrics(Y_test, test_pred);
    end

    % Sample comparison
    fprintf('\nSample Predictions vs Actual (first 10):\n');
    fprintf('Pred | True\n');
    fprintf('-----|-----\n');
    display_samples = min(10, length(train_pred));
    disp([train_pred(1:display_samples), B_train(1:display_samples)]);
end

function print_confusion_matrix(true_labels, predictions)
    labels = unique([true_labels; predictions]);
    cm = confusionmat(true_labels, predictions, 'Order', labels);

    % Print header
    fprintf('     ');
    fprintf('%4d ', labels);
    fprintf('\n');

    % Print matrix
    for i = 1:length(labels)
        fprintf('%4d ', labels(i));
        fprintf('%4d ', cm(i,:));
        fprintf('\n');
    end

    % Calculate and print class-wise accuracy
    class_acc = diag(cm)./sum(cm,2)*100;
    fprintf('\nClass-wise Accuracy:\n');
    for i = 1:length(labels)
        fprintf('Class %d: %.1f%%\n', labels(i), class_acc(i));
    end
end

function print_simple_metrics(true_labels, predictions)
    % Alternative metrics when confusionmat is not available
    labels = unique([true_labels; predictions]);

    fprintf('Class-wise accuracies:\n');
    for i = 1:length(labels)
        class_idx = true_labels == labels(i);
        if any(class_idx)
            class_acc = 100 * sum(predictions(class_idx) == labels(i)) / sum(class_idx);
            fprintf('Class %d: %.1f%%\n', labels(i), class_acc);
        else
            fprintf('Class %d: No samples\n', labels(i));
        end
    end
end
