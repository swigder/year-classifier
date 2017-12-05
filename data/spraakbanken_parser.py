import os
from lxml import etree
from bz2 import BZ2File
import codecs


def process_file(in_file):
    out_files = dict()
    count = 0

    for event, node in etree.iterparse(in_file, tag='text'):
        year = node.attrib['date'][:4]
        text = "".join([x for x in node.itertext()]).replace("\n", " ").strip()
        if year not in out_files:
            out_file = codecs.open('/Users/xx/Downloads/st_data/output/{}.txt'.format(year), 'a', 'utf-8')
            out_files[year] = out_file
        else:
            out_file = out_files[year]
        out_file.write(text + '\n')
        count += 1
        if count % 1000 == 0: print(count)

    for out_file in out_files.values():
        out_file.close()


out_dir = '/Users/xx/Downloads/st_data/input/'
for filename in os.listdir(out_dir):
    try:
        print(filename)
        if filename.endswith(".xml"):
            process_file(out_dir + filename)
        elif filename.endswith(".bz2"):
            with BZ2File(out_dir + filename) as xml_file:
                process_file(xml_file)
    except:
        print('Error!')
    else:
        continue
