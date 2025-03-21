function plotConfusionMatrix(y_true, y_pred, title_str)
    % Create confusion matrix
    conf_mat = confusionmat(y_true, y_pred);

    % Plot confusion matrix
    figure;
    imagesc(conf_mat);
    colorbar;
    title(['Confusion Matrix - ', title_str]);
    xlabel('Predicted Digit');
    ylabel('Actual Digit');

    % Add digit labels
    ax = gca;
    ax.XTickLabel = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
    ax.YTickLabel = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'};
end
