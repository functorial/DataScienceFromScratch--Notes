from bs4 import BeautifulSoup
import requests
from collections import Counter
import sys
import csv

# pass in URL and target file name
url, filename = (sys.argv[1], sys.argv[2])
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

paragraphs = soup.find_all('p')
words = []
for paragraph in paragraphs:
    words += paragraph.text.strip().split()
word_count = Counter(word.lower() for word in words)

with open(filename, 'w', newline='') as n:
    writer = csv.writer(n, delimiter=":")
    writer.writerow(["count", "word"])
    for word, count in word_count.most_common():
        writer.writerow([count, word])
print("\ndone!\n")

        


# for word, count in word_count.most_common(20):
#     print(word, count)

# for link in soup.find_all('a'):
#     print(link.get('href'))
# print("done!")

