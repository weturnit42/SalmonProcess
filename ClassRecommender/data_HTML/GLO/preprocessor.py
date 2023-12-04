import glob
from bs4 import BeautifulSoup
import json

files = glob.glob("*.html")

delimiter = "\t"

for file in files:
    # Make the soup

    with open(file, encoding="utf-8") as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    
    # Filter data
    
    data = json.loads(soup.find(id="__INITIAL_STATE__").string)
    name = data["lecture"]["name"]
    professor = data["lecture"]["professor"]

    reviews = []
    for tag in soup.find_all("div", class_="text"):
        reviews.append(tag.string.replace('\t', '').replace('\n', ''))
    # Save the output

    title = name + "_" + professor + ".tsv"
    with open(title, "w", encoding="utf-8") as fp:
        for review in reviews:
            fp.write(review + '\n')
        print(title + ": finished.")

print("Done.")