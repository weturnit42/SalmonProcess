from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences 
from keras.models import load_model
from konlpy.tag import Okt
from nltk.tokenize import sent_tokenize
import numpy as np

f = open("inYeBonTextv2.txt", "r", encoding='utf-8')
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
model=load_model("inYeBon_epochs400.hdf5") # 모델 불러오기 

wordList="강의".split()
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

    prob = []
    sentences = []
    for j in idx[:n]:
        print(" ".join(wordList[:i]), reverseWordMap[j], "(p={:4.2f}%)".format(100*p[j]))
        prob.append(p[j])
        sentences.append((" ".join(wordList[:i]) + reverseWordMap[j]))

    # return sentences, [prob]
    return (" ".join(wordList[:i]) + " " + reverseWordMap[j])

temp = predictWord(len(wordList),1)

while(True):
    wordList = temp.split()
    if("eos" in wordList):
        break
    temp = predictWord(len(wordList),1)

# def log(number):
#   # log에 0이 들어가는 것을 막기 위해 아주 작은 수를 더해줌.
#   return np.log(number + 1e-10)

# def naive_beam_search_decoder(predictions, k):
#   # prediction = (seq_len , V)
#   sequences = [[list(), 1.0]]
  
#   for row in predictions:
#     print(row)
#     all_candidates = list()
    
#     # 1. 각각의 timestep에서 가능한 후보군으로 확장
#     for i in range(len(sequences)):
#       seq, score = sequences[i]
      
#       # 2. 확장된 후보 스텝에 대해 점수 계산
#       for j in range(len(row)):
#         new_seq = seq + [j]
#         # print(row)
#         new_score = score * -log(row[j])
#         candidate = [new_seq, new_score]
#         all_candidates.append(candidate)
    
# 	# 3. 가능도가 높은 k개의 시퀀스만 남김 
#     ordered = sorted(all_candidates, key=lambda tup:tup[1]) #점수 기준 정렬
#     sequences = ordered[:k]
    
#   return sequences

# import random
# seq_len = 2
# V = 5
# predictions = [[random.random() for _ in range(V)] for _ in range(seq_len)]
# print(predictions)
# print(naive_beam_search_decoder(predictWord(1,2)[1], 2))