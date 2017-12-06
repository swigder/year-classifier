import argparse
import os
from collections import defaultdict

from lxml import etree
from bz2 import BZ2File
import codecs


def process_file(in_file, name):
    out_files = dict()
    count = 0

    for event, elem in etree.iterparse(in_file, tag='text'):
        try:
            year = elem.attrib['date'][:4]
        except:
            year = elem.attrib['year']
        if year not in out_files:
            out_file = codecs.open('/{}-{}.txt'.format(out_dir, year, name), 'w', 'utf-8')
            out_files[year] = out_file
            out_file.write(year + '\n')
        else:
            out_file = out_files[year]
        for sentence in elem:
            text = "".join([x for x in sentence.itertext()]).replace("\n", " ").strip()
            out_file.write(text + '\n')
            count += 1
            if count % 50000 == 0:
                print(name, count)
        elem.clear()
        for ancestor in elem.xpath('ancestor-or-self::*'):
            while ancestor.getprevious() is not None:
                del ancestor.getparent()[0]

    for out_file in out_files.values():
        out_file.close()


parser = argparse.ArgumentParser(description='Parse GU Corpora.')
parser.add_argument('dir', type=str, help='directory with input and output subdirectories')
parser.add_argument('-p', '--parse', action='store_true')
parser.add_argument('-s', '--stats', action='store_true')

args = parser.parse_args()

in_dir = args.dir + '/input/'

if args.parse:
    for filename in os.listdir(in_dir):
        try:
            print(filename)
            nice_name = filename.split('.')[0]
            if filename.endswith(".xml"):
                process_file(in_dir + filename, nice_name)
            elif filename.endswith(".bz2"):
                with BZ2File(in_dir + filename) as xml_file:
                    process_file(xml_file, nice_name)
        except Exception as e:
            print('Error!', e)
        else:
            continue

out_dir = args.dir + '/output/'

if args.stats:
    import matplotlib.pyplot as plt

    years = defaultdict(int)
    for filename in os.listdir(out_dir):
        if not filename.endswith('.txt'):
            continue
        print(filename)
        with codecs.open(out_dir + filename, 'r', 'utf-8') as file:
            year = int(file.readline())
            num_lines = sum(1 for line in file)
            years[year] += num_lines
    plt.bar(list(years.keys()), years.values(), color='g')
    plt.yscale('log', nonposy='clip')
    plt.show()

