import argparse
import pandas as pd
import matplotlib.pyplot as plt

import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='Clean up.')
parser.add_argument('dir', type=str, help='directory with data files')
parser.add_argument('out', type=str, help='directory for report')

args = parser.parse_args()

corpora = defaultdict(dict)
decades = set()

for filename in os.listdir(args.dir):
    if not filename.endswith(".txt"):
        continue
    decade = int(filename[:3] + '0')
    decades.add(decade)
    corp = ''.join(filter(str.isalpha, filename[5:filename.index('.')]))
    if corp.startswith('kubhist'):
        corp = corp[len('kubhist'):]
    corpora[corp][decade] = sum(1 for line in open(os.path.join(args.dir, filename)))

table = pd.DataFrame(columns=sorted(decades))

for corpus, hits in corpora.items():
    for decade, count in hits.items():
        table.at[corpus, decade] = count
table.loc['Total'] = table.sum(axis=0)

table.to_csv(os.path.join(args.out, 'report.csv'))

# for corpus, hits in corpora.items():
#     plt.plot(*zip(*sorted(hits.items())), label=corpus)
# plt.plot(decades, table.loc['Total'].values.flatten().tolist(), 'Total')
# plt.legend()
# plt.show()
