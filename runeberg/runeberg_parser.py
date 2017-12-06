from bs4 import BeautifulSoup
import re
import os

def parse_file(path):
    fn=re.search('\_(\d{4})\.txt', path)
    year=fn.group(1)

    with open(path) as fp:
        soup = BeautifulSoup(fp, "lxml")

    all_text=''
    for text in soup.find_all('chapter'):
        #print(text.get('date'))
        t=(text.get_text().replace("\n", " "))
        all_text+=t

    s=re.split('(?<=\.)\s+(?=[A-ZÅÄÖ])', all_text)
    #print(s[:100])
    #print(len(s))
    return s, year

def save_sentences(path,sentences, year):
    with open(path, 'w+t') as f:
        f.write(year+"\n")
        f.write("\n".join(sentences))

folder_path='data/raw'
result_folder_path='data/formated'
dirs=os.listdir(folder_path)
print(dirs)
for filename in dirs:
    # Read and parse file. Get back a list of sentences
    path='{}/{}'.format(folder_path,filename)
    sentences, year=parse_file(path)

    print('name: {}, year: {}, #sentences: {}'.format(filename, year, len(sentences)))
    path='{}/{}'.format(result_folder_path,filename)
    save_sentences(path, sentences, year)

