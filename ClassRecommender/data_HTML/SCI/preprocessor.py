import glob
from bs4 import BeautifulSoup
import json

files = glob.glob("*.html")

print(len(files))

delimiter = "\t"

for file in files:
    # Make the soup

    with open(file, encoding="utf-8") as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    
    # Filter data
    
    data = json.loads(soup.find(id="__INITIAL_STATE__").string)
    name = data["lecture"]["name"]
    professor = data["lecture"]["professor"]

    reviews = [tag.string.replace(delimiter, " ") for tag in soup.find_all("div", class_="text")]

    # Save the output

    title = name + "_" + professor + ".tsv"
    with open(title, "a", encoding="utf-8") as fp:
        fp.write(delimiter.join(reviews))
        print(title + ": finished.")

print("Done.")