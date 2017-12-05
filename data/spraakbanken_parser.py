import argparse
import os
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
            out_file = codecs.open('{}/output/{}-{}.txt'.format(args.dir, year, name), 'w', 'utf-8')
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

args = parser.parse_args()

in_dir = args.dir + '/input/'
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
