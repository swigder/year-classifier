import argparse
import codecs
import os


def get_valid_line(file_reader):
    line = file_reader.readline()
    while line is not '' and (len(line) < args.sample_min or len(line) > args.sample_max):
        line = file_reader.readline()
    return line


def generate_dataset(readers, samples, interval_test):
    training = []
    test = []
    i = 0
    remaining_files = list(readers)
    exhausted_files = []  # avoid modifying remaining_files while iterating over it
    while len(test) < samples / interval_test and len(remaining_files) > 0:
        for file_reader in remaining_files:
            line = get_valid_line(file_reader)
            if line == '':
                exhausted_files.append(file_reader)
                continue
            if i % interval_test == (interval_test - 1):
                test.append(line)
            else:
                training.append(line)
        i += 1
        remaining_files = [file for file in remaining_files if file not in exhausted_files]

    if len(remaining_files) == 0:
        print('Exhausted dataset!')

    print(len(training), len(test), len(training) + len(test), '\n')
    return training, test


def write_to_file(dir, lines, period):
    with codecs.open("{}/{}0.txt".format(dir, period), 'w', 'utf-8') as file:
        file.write("{}0\n".format(period))
        file.writelines(lines)


parser = argparse.ArgumentParser(description='Clean up.')
parser.add_argument('dir', type=str, help='directory with data files')
parser.add_argument('-s', '--samples', type=int, default=5000, help='number of samples per decade')
parser.add_argument('-t', '--test', type=int, default=5, help='1/n test')
parser.add_argument('-f', '--first', type=int, default=177, help='first decade, excluding final 0')
parser.add_argument('-l', '--last', type=int, default=201, help='first decade, excluding final 0')
parser.add_argument('-a', '--automatic_period', action='store_true')
parser.add_argument('-p', '--period', type=int, default=1, help='decades per period')
parser.add_argument('-smin', '--sample_min', type=int, default=100, help='min chars per sample')
parser.add_argument('-smax', '--sample_max', type=int, default=2000, help='max chars per sample')

args = parser.parse_args()


dataset_dir = "{}/dataset-{}-{}-{}-{}".format(args.dir, args.period, args.samples, args.sample_min, args.sample_max)
training_dir = "{}/training".format(dataset_dir)
test_dir = "{}/test".format(dataset_dir)
os.makedirs(training_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

files = [filename for filename in os.listdir(args.dir) if filename.endswith('.txt')]
first_decade, last_decade = \
    (args.first, args.last) if not args.automatic_period \
    else (int(files[0][:3]), int(files[-1][:3]))

for period in range(first_decade, last_decade + 1, args.period):
    decade_files = []
    for decade in range(period, period+args.period):
        decade_files += [filename for filename in files if filename.startswith(str(decade))]
    file_readers = []
    for filename in decade_files:
        path = "{}/{}".format(args.dir, filename)
        file = codecs.open(path, 'r', 'utf-8')
        file.readline()  # get rid of 'year' line
        file_readers.append(file)
        # print('Reading file {}'.format(filename))
    print('Processing period {}0...'.format(period))
    training_sentences, test_sentences = generate_dataset(file_readers, args.samples * args.period, args.test)
    for file in file_readers:
        file.close()

    write_to_file(training_dir, training_sentences, period)
    write_to_file(test_dir, test_sentences, period)
