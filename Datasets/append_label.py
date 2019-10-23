with open('temp', 'r') as f:
    lines = []
    for line in f.readlines():
        lines.append(line[:-1])

with open('temp', 'w') as f:
    for line in lines:
        f.write(line+'\tNegative\n')
