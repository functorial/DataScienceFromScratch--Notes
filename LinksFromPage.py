from bs4 import BeautifulSoup
import requests
from collections import Counter
import sys
import csv

# pass in URL and target file name
url, filename = (sys.argv[1], sys.argv[2])

html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

links = [a['href'] for a in soup('a') if a.has_attr('href')]
link_count = Counter(links).most_common()

with open(filename, 'w', newline='') as n:
    writer = csv.writer(n, delimiter=":")
    writer.writerow(["count", "link"])
    for link, count in link_count:
        writer.writerow([count, link])

print("\ndone!\n")

        
# for link in soup.find_all('a'):
#     print(link.get('href'))