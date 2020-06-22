from random import random
import csv

n1 = 0 / 66245
n2 = 500 / 15083
datadirectory = "/Users/jik/data/trec-podcasts/"
infile = "sampledescriptions-many-of-them.tsv"
notherinfile = "opposite.sampledescriptions-many-of-them.tsv"
outfile = "sampledescriptions-smallset.tsv"

output = []
with open(datadirectory + infile, 'r') as infile:
    for row in infile:
        if random() < n1:
            output.append(row)
print(len(output))
with open(datadirectory + notherinfile, 'r') as infile:
    for row in infile:
        if random() < n2:
            output.append(row)
print(len(output))
with open(datadirectory + outfile, "w") as outfile:
    for o in output:
        outfile.write(o)
