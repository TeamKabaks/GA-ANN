function convert_digit_file(input_filename, output_filename)
    % Open files for reading and writing
    infile = fopen(input_filename, "r");
    outfile = fopen(output_filename, "w");

    if infile == -1
        error("Could not open input file.");
    endif
    if outfile == -1
        error("Could not open output file.");
    endif

    matrix = {}; % Store the current 32x32 matrix
    while ~feof(infile)
        line = strtrim(fgetl(infile)); % Read line and remove leading/trailing spaces

        if isdigit(line) && length(line) == 1  % If it's a single digit, it's a label
            if ~isempty(matrix)  % Ensure a matrix was collected before writing
                for i = 1:length(matrix)
                    fprintf(outfile, "%s\n", strjoin(matrix{i}, ",")); % Write matrix row with commas
                end
                fprintf(outfile, "%s\n", line); % Write label after matrix
                matrix = {};  % Reset for the next matrix
            endif
        else
            matrix{end+1} = cellstr(line(:)'); % Convert binary string into a cell array
        endif
    endwhile

    % Close files
    fclose(infile);
    fclose(outfile);

    fprintf("Converted file saved as %s\n", output_filename);
end
