from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import load_model
from konlpy.tag import Okt
from nltk.tokenize import sent_tokenize
import numpy as np

f = open("AACText.txt", "r", encoding='utf-8')
c = f.read()
f.close()

okt = Okt()
doc0=[" ".join( ["".join(w) for w , t in okt.pos(s)] ) for s in sent_tokenize(c) ] 
   
# 텍스트 제너레이션
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 

tokenizer = Tokenizer()
tokenizer.fit_on_texts(doc0)
doc = [ tok for tok in tokenizer.texts_to_sequences(doc0) if len(tok) > 1 ]  # 공백문자 제거를 위해 

maxLen = max([len(x)-1 for x in doc] ) # 0번부터 시작하게 하려고 
print(maxLen)
vocabSize = len(tokenizer.word_index)+1 #1164 + 1 
print(vocabSize)
model=load_model("AAC.hdf5") # 모델 불러오기 

wordList="이 강의 는".split()
reverseWordMap = dict(map(reversed,tokenizer.word_index.items()))

x = pad_sequences( [[tokenizer.word_index[w] for w in wordList[:2] ]] , maxlen=maxLen)

p = model.predict(x)[0]

idx = np.flip(p.argsort(),0 ) # 내림차순 정렬

# for i in idx[:10]:
#     print(reverseWordMap[i])

def predictWord(i, n):
    x = pad_sequences([[tokenizer.word_index[w] for w in wordList[:i]]], maxlen = maxLen)
    p = model.predict(x)[0]
    idx = np.flip(np.argsort(p) , 0)
    for j in idx[:n]:
        print(" ".join(wordList[:i]), reverseWordMap[j], "(p={:4.2f}%)".format(100*p[j]))

    return (" ".join(wordList[:i]) + " " + reverseWordMap[j])

temp = predictWord(len(wordList),1)

while(True):
    wordList = temp.split()
    if("eos" in wordList):
        break
    temp = predictWord(len(wordList),1)