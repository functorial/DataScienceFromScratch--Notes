from bs4 import BeautifulSoup
import requests
url = "https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

paragraphs = soup('p')
paragraphs_text = [p.text for p in paragraphs]

# p.attrs is a dict
# has 'class' key, with id a list of paragraph tag classes
# in practice, need to reason using html file for classes
important_paragraphs =  [p for p in paragraphs if 'important' in p.attrs.get('class')]

# or you could do
important_paragraphs2 = soup('p', 'important')