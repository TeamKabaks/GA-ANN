function [X, y] = loadDigitData(filename)
    % Read the file as text
    fileContent = fileread(filename);
    % Split into lines
    lines = strsplit(fileContent, '\n');

    % Find where the digit data starts (after metadata)
    dataStartLine = 0;
    for i = 1:length(lines)
        if length(lines{i}) == 32 && all(ismember(lines{i}, '01'))
            dataStartLine = i;
            break;
        end
    end

    if dataStartLine == 0
        error('Could not find the start of digit data');
    end

    % Determine how many digits we have
    remainingLines = length(lines) - dataStartLine + 1;
    numDigits = floor(remainingLines / 33); % Each digit has 32 lines + 1 label line

    % Initialize arrays
    X = zeros(numDigits, 32*32); % Each row is a flattened 32x32 image
    y = zeros(numDigits, 1);     % Labels

    % Process each digit
    for i = 1:numDigits
        startLine = dataStartLine + (i-1)*33;
        pixelLines = lines(startLine:startLine+31);

        % Extract pixel data
        pixelData = [];
        for j = 1:32
            if j <= length(pixelLines) && ~isempty(pixelLines{j})
                line = pixelLines{j};
                % Convert the line to a numeric array of 0s and 1s
                row = arrayfun(@(c) str2double(c), line);
                pixelData = [pixelData, row]; % Append the row
            end
        end

        % Store in X
        X(i, :) = pixelData(:)'; % Ensure the row is 1x1024

        % Get the label
        labelLine = lines{startLine+32};
        if ~isempty(labelLine)
            labelLine = strtrim(labelLine);
            y(i) = str2double(labelLine);
            if isnan(y(i))
                warning('Invalid label for digit %d. Setting to 0.', i);
                y(i) = 0;
            end
        end
    end

        % Normalize X using Min-Max scaling
    X = (X - min(X(:))) / (max(X(:)) - min(X(:)));

    % Debugging: Check normalization range
    fprintf('X Min: %f, X Max: %f\n', min(X(:)), max(X(:)));


    fprintf('Loaded %d samples of 32x32 digits\n', numDigits);
end
