import argparse
import codecs

import os

import re


def clean_too_long(line):
    if len(line) < 1000:
        return False, line
    split = re.split('(?<=\.)\s+', line)
    return len(split) > 1, split


def clean_junk(line):
    alpha = sum([1 for char in line if char.isalpha()])
    if alpha > len(line) / 2:
        return False, [line]
    return True, []


def process_file(dir, filename, old_data_dir, clean_fn):
    path = '{}/{}'.format(dir, filename)
    with codecs.open(path, 'r', 'utf-8') as file:
        will_clean = False
        output_lines = [file.readline()]
        for line in file.readlines():
            is_dirty_line, clean_lines = clean_fn(line)
            output_lines += clean_lines
            will_clean |= is_dirty_line
    if will_clean:
        print('Need to rewrite file {}!'.format(filename))
        os.rename(path, old_data_dir + filename)
        with codecs.open(path, 'w', 'utf-8') as file:
            file.writelines(output_lines)


fns = {'junk': clean_junk, 'long': clean_too_long}


parser = argparse.ArgumentParser(description='Clean up.')
parser.add_argument('dir', type=str, help='directory with data files')
parser.add_argument('-t', '--cleanup_type', choices=fns.keys(), help='type of cleanup')

args = parser.parse_args()

old_data_dir = args.dir + "/{}/".format(args.cleanup_type)
os.makedirs(old_data_dir, exist_ok=True)

for filename in os.listdir(args.dir):
    if filename.endswith(".txt"):
        try:
            print(filename)
            nice_name = filename.split('.')[0]
            if filename.endswith(".txt"):
                process_file(args.dir, filename, old_data_dir, fns[args.cleanup_type])
        except Exception as e:
            print('Error!', e)
    else:
        continue
