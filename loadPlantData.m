function [X, y] = loadPlantData(filename)
% LOADPLANTDATA Loads and preprocesses plant distribution data
% Avoids all functions that might trigger UTF-8 encoding errors
fprintf('\nReading data from file: %s', filename);

% Read the file as plain text
fid = fopen(filename, 'r');
if fid == -1
    error('Could not open file %s', filename);
end

fprintf('\nReading each line...');
% Read each line
lines = {};
lineNum = 1;
while ~feof(fid)
    tline = fgetl(fid);
    if ischar(tline)
        lines{lineNum} = tline;
        lineNum = lineNum + 1;
        fprintf('\nLine %d: %s', lineNum-1, tline);
    end
end
fclose(fid);
fprintf('\nFinished reading %d lines', length(lines));

% Parse the data manually without using strsplit
fprintf('\nParsing data into plants and locations...');
plants = {};
locations = {};
numPlants = 0;

fprintf('\nBeginning line-by-line parsing:');
for i = 1:length(lines)
    line = lines{i};
    if isempty(line)
        fprintf('\nLine %d: Empty line - skipping', i);
        continue;
    end

    % Find all commas
    commaPositions = strfind(line, ',');
    if isempty(commaPositions)
        fprintf('\nLine %d: No commas found - skipping', i);
        continue;
    end

    % Extract plant name (before first comma)
    plantName = line(1:commaPositions(1)-1);

    % Extract locations (between commas)
    currentLocations = {};
    for j = 1:length(commaPositions)
        startPos = commaPositions(j) + 1;
        if j < length(commaPositions)
            endPos = commaPositions(j+1) - 1;
        else
            endPos = length(line);
        end

        if startPos <= endPos
            location = line(startPos:endPos);
            if ~isempty(location)
                currentLocations{end+1} = location;
            end
        end
    end

    if ~isempty(currentLocations)
        numPlants = numPlants + 1;
        plants{numPlants} = plantName;
        locations{numPlants} = currentLocations;
        fprintf('\nLine %d: Added plant "%s" with %d locations', i, plantName, length(currentLocations));
    else
        fprintf('\nLine %d: No locations found for plant "%s" - skipping', i, plantName);
    end
end
fprintf('\nExtracted %d plants with location data', numPlants);

% Get all unique locations
fprintf('\nIdentifying unique locations...');
allLocations = {};
locationCount = 0;

for i = 1:length(locations)
    for j = 1:length(locations{i})
        loc = locations{i}{j};
        % Check if this location is already in our list
        found = false;
        for k = 1:length(allLocations)
            if strcmp(loc, allLocations{k})
                found = true;
                break;
            end
        end
        if ~found
            allLocations{end+1} = loc;
            locationCount = locationCount + 1;
            fprintf('\nFound new location #%d: %s', locationCount, loc);
        end
    end
end
fprintf('\nIdentified %d unique locations', length(allLocations));

% Create feature matrix
fprintf('\nCreating feature matrix (%d plants × %d locations)...', length(plants), length(allLocations));
numLocations = length(allLocations);
X = zeros(length(plants), numLocations);
nonzeroCount = 0;

for i = 1:length(plants)
    plantLocCount = 0;
    for j = 1:length(locations{i})
        loc = locations{i}{j};
        % Find the index of this location
        for k = 1:length(allLocations)
            if strcmp(loc, allLocations{k})
                X(i, k) = 1;
                plantLocCount = plantLocCount + 1;
                nonzeroCount = nonzeroCount + 1;
                break;
            end
        end
    end
    fprintf('\nPlant %d (%s): Found in %d locations', i, plants{i}, plantLocCount);
end
fprintf('\nFeature matrix created with %d nonzero entries (%.2f%% density)', nonzeroCount, (nonzeroCount/(length(plants)*numLocations))*100);

% Extract genus names manually
fprintf('\nExtracting genus names...');
genera = {};
for i = 1:length(plants)
    plantName = plants{i};
    % Find the first space
    spacePos = strfind(plantName, ' ');
    if isempty(spacePos)
        % If no space, the entire name is the genus
        genus = plantName;
        fprintf('\nPlant %d (%s): No species name found, using entire name as genus', i, plantName);
    else
        % Otherwise, the genus is the text before the first space
        genus = plantName(1:spacePos(1)-1);
        fprintf('\nPlant %d (%s): Extracted genus "%s"', i, plantName, genus);
    end
    genera{i} = genus;
end

% Get unique genera
fprintf('\nIdentifying unique genera...');
uniqueGenera = {};
genusCount = 0;

for i = 1:length(genera)
    % Check if this genus is already in our list
    found = false;
    for j = 1:length(uniqueGenera)
        if strcmp(genera{i}, uniqueGenera{j})
            found = true;
            break;
        end
    end
    if ~found
        uniqueGenera{end+1} = genera{i};
        genusCount = genusCount + 1;
        fprintf('\nFound new genus #%d: %s', genusCount, genera{i});
    end
end
fprintf('\nIdentified %d unique genera', length(uniqueGenera));

% Create label vector
fprintf('\nCreating label vector...');
y = zeros(length(plants), 1);
for i = 1:length(plants)
    for j = 1:length(uniqueGenera)
        if strcmp(genera{i}, uniqueGenera{j})
            y(i) = j;
            fprintf('\nPlant %d (%s): Assigned to genus %d (%s)', i, plants{i}, j, uniqueGenera{j});
            break;
        end
    end
end

fprintf('\n=== SUMMARY ===');
fprintf('\nLoaded %d plants with %d locations and %d genera\n', ...
    length(plants), numLocations, length(uniqueGenera));
fprintf('Feature matrix size: %d × %d\n', size(X, 1), size(X, 2));
fprintf('Label vector size: %d × 1\n', length(y));
fprintf('Memory usage estimate: %.2f KB\n', (numel(X) + numel(y)) * 8 / 1024);
end
