import argparse
import codecs
import os


def generate_dataset(line_sets, samples, interval_test):
    training = []
    test = []
    i = 0
    available_sets = list(line_sets)
    while len(test) < samples / interval_test and len(available_sets) > 0:
        for line_set in available_sets:
            if i % interval_test == (interval_test - 1):
                test.append(line_set[i])
            else:
                training.append(line_set[i])
        i += 1
        available_sets = [line_set for line_set in available_sets if len(line_set) > i]

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
parser.add_argument('-p', '--period', type=int, default=1, help='decades per period')

args = parser.parse_args()


training_dir = "{}/training-{}".format(args.dir, args.period)
test_dir = "{}/test-{}".format(args.dir, args.period)
os.makedirs(training_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

files = [filename for filename in os.listdir(args.dir) if filename.endswith('.txt')]

for period in range(args.first, args.last + 1, args.period):
    decade_files = []
    for decade in range(period, period+args.period):
        decade_files += [filename for filename in files if filename.startswith(str(decade))]
    lines = []
    total = 0
    for filename in decade_files:
        path = args.dir + filename
        with codecs.open(path, 'r', 'utf-8') as file:
            file.readline()
            lines.append(list(file.readlines()))
        print('Read {} lines in file {}'.format(len(lines[-1]), filename))
        total += len(lines[-1])
    print('Period {}0 total: {}\n'.format(period, total))
    training_sentences, test_sentences = generate_dataset(lines, args.samples * args.period, args.test)

    write_to_file(training_dir, training_sentences, period)
    write_to_file(test_dir, test_sentences, period)
