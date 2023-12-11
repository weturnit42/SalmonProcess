import glob
from bs4 import BeautifulSoup
import json

files = glob.glob("*.html")
d, i = {}, 0

for file in files:
    
    # Make the soup
    
    with open(file, encoding='utf-8') as fp:
        soup = BeautifulSoup(fp, 'html.parser')
    
    # Filter data
    
    data = json.loads(soup.find(id="__INITIAL_STATE__").string)
    label = data["lecture"]["name"] + "_" + data["lecture"]["professor"]
    
    if label in d: continue
    d[label] = label_num = str(i); i += 1

    reviews = [tag.string.replace("\n", " ").replace("\t", " ") for tag in soup.find_all("div", class_="text")]
    
    # Save the output

    with open("data.tsv", "a", encoding='utf-8') as fp:
        for review in reviews:
            fp.write(review + "\t" + label_num + "\n")
    
    with open("dict.tsv", "a", encoding='utf-8') as fp:
        fp.write(label_num + "\t" + label + "\n")
    
    print(label + ": finished.")

print("Done.")
