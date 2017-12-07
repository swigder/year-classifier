import argparse
import codecs

import os


def is_junk(i, line):
    if len(line) < 2:
        return True
    alpha = sum([1 for char in line if char.isalpha()])
    return i > 0 and alpha < len(line) / 2


def process_file(dir, filename, junkdir):
    path = '{}/{}'.format(dir, filename)
    with codecs.open(path, 'r', 'utf-8') as file:
        found_junk = False
        non_junk_lines = []
        for i, line in enumerate(file.readlines()):
            if not is_junk(i, line):
                non_junk_lines.append(line)
            else:
                found_junk = True
    if found_junk:
        print('Found junk at {}!'.format(filename))
        os.rename(path, junkdir + filename)
        with codecs.open(path, 'w', 'utf-8') as file:
            file.writelines(non_junk_lines)


parser = argparse.ArgumentParser(description='Remove junk.')
parser.add_argument('dir', type=str, help='directory with input and output subdirectories')

args = parser.parse_args()

junkdir = args.dir + "/junk/"
os.makedirs(junkdir, exist_ok=True)

for filename in os.listdir(args.dir):
    if filename.endswith(".txt"):
        try:
            print(filename)
            nice_name = filename.split('.')[0]
            if filename.endswith(".txt"):
                process_file(args.dir, filename, junkdir)
        except Exception as e:
            print('Error!', e)
    else:
        continue
