from bs4 import BeautifulSoup
import requests
import re
from typing import Dict, Set

url = "https://house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, 'html5lib')

print(f"Gathering all links from {url}...")
all_urls = [a['href'] for a in soup('a') if a.has_attr('href')]

# only interested in certain urls
# ^ means start of string
# $ means end of string
# ? means one or none of the preceding character
# * means none or more of the preceding character
# . means any character except line break, so .* means any number of any non-line break characters
# \ is an escape character, so \. just means period
regex = r"^https?://.*\.house\.gov/?$"
print(f"Filtering good links from all links...")
good_urls = [url for url in all_urls if re.match(regex, url)]

# A way to remove duplicates
good_urls = list(set(good_urls))

# if you look, most of the sites have a link to press releases
# html2 = requests.get("https://kustoff.house.gov").text
# soup2 = BeautifulSoup(html2, 'html5lib')

# use a set because the links might appear multiple times
#links = {a['href'] for a in soup2('a') if ('press releases' in a.text.lower() and a.has_attr('href'))}
# print(links)

# linksb = set()
# for a in soup2('a'):
#     print(a)
#     if ('press releases' in a.text.lower() and a.has_attr('href')):
#         linksb.add(a['href'])
#         print(f"a: {a}")
#         print(f"a.text: {a.text}")
#         print(f"a.text.lower(): {a.text.lower()}")
# print(f"linksb: {linksb}")

press_releases: Dict[str, Set[str]] = {}

print("Gathering press release links from good links...")
for house_url in good_urls:
    html = requests.get(house_url).text
    soup3 = BeautifulSoup(html, 'html5lib')
    press_release_links = {a['href'] for a in soup3('a') if ('press releases' in a.text.lower() and a.has_attr('href'))}
    press_releases[house_url] = press_release_links

# let's filter our results for press releases containing a specific keyword!
def paragraph_mentions(text: str, keyword: str) -> bool:
    """
    Returns True if a <p> inside the text mentions {keyword}
    """
    soup = BeautifulSoup(text, 'html5lib')
    paragraphs = [p.get_text() for p in soup('p')]

    return any(keyword.lower() in paragraph.lower() for paragraph in paragraphs)

# let's test it out...
text1 = "<body><h1>Facebook</h1><p>Twitter</p>"
assert paragraph_mentions(text1, "Twitter")
assert not paragraph_mentions(text1, "Facebook")

# Time to scrape!
keyword = 'data'
print(f"Will now print out links which mention the keyword: {keyword}")
for house_url, pr_links in press_releases.items():
    for pr_link in pr_links:
        url = f"{house_url}/{pr_link}"              # `pr_link` is a relative link!
        text = requests.get(url).text

        if paragraph_mentions(text, keyword):
            print(url)
            break

print("Done!")