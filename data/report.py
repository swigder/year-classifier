import argparse
import pandas as pd
import matplotlib.pyplot as plt

import os
from collections import defaultdict

parser = argparse.ArgumentParser(description='Clean up.')
parser.add_argument('dir', type=str, help='directory with data files')
parser.add_argument('out', type=str, help='directory for report')

args = parser.parse_args()

corpora_nice_names = {'aftonbladet': 'Aftonbladet',
                      'attasidor': '8 SIDOR',
                      'blekingsposten': 'Blekingsposten',
                      'bollnastidning': 'Bollnäs tidning',
                      'dalpilen': 'Dalpilen',
                      'dn': 'Dagens Nyheter',
                      'fahluweckoblad': 'Fahlu weckoblad',
                      'goteborgsweckoblad': 'Goteborgs weckoblad',
                      'gotlandstidning': 'Gotlands tidning',
                      'gp': 'Göteborgsposten',
                      'kalmar': 'Kalmar',
                      'ordat': 'Ordat',
                      'postochinrikestidning': 'Post- och inrikes tidning,',
                      'press': 'Press',
                      'runebergdiverse': 'Runeberg diverse tidningar',
                      'ubkvtdagny': 'Dagny',
                      'ubkvtidun': 'Idun',
                      'webbnyheter': 'Webbnyheter',
                      }

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
    corp = corpora_nice_names[corp]
    if decade not in corpora[corp]:
        corpora[corp][decade] = 0
    corpora[corp][decade] += sum(1 for line in open(os.path.join(args.dir, filename)))

table = pd.DataFrame(columns=sorted(decades))

for corpus, hits in corpora.items():
    for decade, count in hits.items():
        table.at[corpus, decade] = count
table.loc['Total'] = table.sum(axis=0)

pd.options.display.float_format = '{:,.0f}'.format

print(table[list(range(1770, 1830, 10))].dropna(how='all').to_csv(float_format='%.0g'))
print(table[list(range(1830, 1890, 10))].dropna(how='all').to_csv(float_format='%.0g'))
print(table[list(range(1890, 1950, 10))].dropna(how='all').to_csv(float_format='%.0g'))
print(table[list(range(1950, 2020, 10))].dropna(how='all').to_csv(float_format='%.0g'))

# table.to_csv(os.path.join(args.out, 'report.csv'))

# for corpus, hits in corpora.items():
#     plt.plot(*zip(*sorted(hits.items())), label=corpus)
# plt.plot(decades, table.loc['Total'].values.flatten().tolist(), 'Total')
# plt.legend()
# plt.show()
