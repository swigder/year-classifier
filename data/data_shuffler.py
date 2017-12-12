import argparse

import os
import random

parser = argparse.ArgumentParser(description='Clean up.')
parser.add_argument('dir', type=str, help='directory with data files')

args = parser.parse_args()

shuffled_dir = args.dir + "/../shuffled/"
print(shuffled_dir)
os.makedirs(shuffled_dir, exist_ok=True)

for filename in os.listdir(args.dir):
    if not filename.endswith(".txt"):
        continue
    try:
        print(filename)
        with open(os.path.join(args.dir, filename), 'r') as in_file:
            year = in_file.readline()
            lines = list(in_file.readlines())
            random.shuffle(lines)
            with open(os.path.join(shuffled_dir, filename), 'w') as out_file:
                out_file.write(year)
                out_file.writelines(lines)
    except Exception as e:
        print('Error!', e)
