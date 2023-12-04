from bs4 import BeautifulSoup

response = open("AACHTML.txt", "r", encoding='utf-8')
soup = BeautifulSoup(response, "html.parser")

reviewCount = soup.select("body > div > div > div.pane > div > div.articles")[0]
reviewTotal = ''

for i in list(range(1, len(reviewCount)-1)):
    print(i)
    # body > div > div > div.pane > div > div.articles > div:nth-child(1) > div.text
    text = reviewCount.select('div:nth-child(' + str(i) + ') > div.text')[0].get_text()
    reviewTotal = reviewTotal + text + ' <eos> '

f = open("AACText.txt", "w", encoding='utf-8')
f.write(reviewTotal)
f.close()