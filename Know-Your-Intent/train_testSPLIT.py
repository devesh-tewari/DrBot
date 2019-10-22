# lines = []
# with open('temp', 'r') as raw:
#     for line in raw.readlines():
#         lines.append(line)
#
# with open('temp', 'w') as raw:
#     for line in lines:
#         raw.write(line.strip('\n')+"\tSymptom/Disease\n")


import pandas
data = pandas.read_csv('raw.tsv', sep='\t', header=None)

# consult = data[data[1] == 'Consult']
# sd = data[data[1] == 'Symptom/Disease']

from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2)

# print(train.head())
# print(test.head())
# print(len(train))
# print(len(test))

train.to_csv('train.tsv', sep='\t', index=None, header=None)
test.to_csv('test.tsv', sep='\t', index=None, header=None)
