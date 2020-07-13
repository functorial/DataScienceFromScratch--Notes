import requests, json
from collections import Counter
from dateutil.parser import parse

github_user = "joelgrus"
endpoint = f"https://api.github.com/users/{github_user}/repos"

# a list of dictionaries, each representing a repo
repos = json.loads(requests.get(endpoint).text)


# collecting date/time of creations
print(parse(repos[0]["created_at"]))
print(parse(repos[0]["created_at"]).month)
print(parse(repos[0]["created_at"]).day)
print(parse(repos[0]["created_at"]).year)

# all dates
dates = [parse(repo["created_at"]) for repo in repos]

last_5_repos = sorted(repos, key=lambda r: r["pushed_at"], reverse=True)[:5]
last_5_languages = [repo["language"] for repo in last_5_repos]
print(last_5_languages)