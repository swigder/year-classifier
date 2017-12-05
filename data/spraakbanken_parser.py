from bs4 import BeautifulSoup

with open("index.html") as fp:
    soup = BeautifulSoup(fp)

for text in soup.find_all('text'):
    print(text.get('date'))
    print(text.get_text().replace("\n", " "))
