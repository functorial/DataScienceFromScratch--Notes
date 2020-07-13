from bs4 import BeautifulSoup
import requests
from collections import Counter

url = ("https://en.wikipedia.org/wiki/Final_Fantasy")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

links = soup.find_all('a')
paragraphs = soup.find_all('p')
words = []
for paragraph in paragraphs:
    words += paragraph.text.split()

c_words = Counter(word.lower() for word in words)

for word, count in c_words.most_common(20):
    print(word, count)

# for link in soup.find_all('a'):
#     print(link.get('href'))
# print("done!")

