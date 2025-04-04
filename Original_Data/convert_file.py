filepath = "C:/Users/IDEAPAD/Documents/GitHub/GA-ANN/Original_Data/"

input_filename = filepath + "writer-dependent.txt"
output_filename = filepath + "writer-dependent.dat"

with open(input_filename, "r") as infile, open(output_filename, "w") as outfile:
    matrix = []  # Store the current 32x32 matrix
    for line in infile:
        line = line.strip()
        if line.isdigit() and len(line) == 1:  # If single digit, it's a label
            if matrix:
                for row in matrix:
                    outfile.write(",".join(row) + "\n")  # write matrix row
                outfile.write(line + "\n")  # write label after matrix
                matrix = []  # reset for the next matrix
        else:
            matrix.append(list(line))  # binary string to list

print(f"Converted file saved as {output_filename}")
