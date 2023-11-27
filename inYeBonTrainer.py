# text generation 
# from konlpy.corpus import kolaw # 헌법데이터
from konlpy.tag import Okt# 형태소 분석기 
from nltk.tokenize import sent_tokenize

# c = kolaw.open("constitution.txt").read()
# # c # len(18884)

f = open("inYeBonTextv2.txt", "r", encoding='utf-8')
c = f.read()
f.close()
# c = "과제가 귀찮지만 이만큼 할거 없는 수업도 없는듯 중간 과제 개꿀입니다 30분안에 할수있어요 매주 보고서 쓰는것도 없어서 좋습니다 한양대 인정 꿀명강의. 출석잘하고 과제 잘 제출하면 됩니다 과제와 출석만 잘 하면 됩니다 영역 채우기 좋은 교양강의"
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

from keras.utils import to_categorical

def genData(doc, maxLen, vocabSize):
    for sent in doc:
        inputs=[]
        targets=[]
        for i in range(1, len(sent)): 
            inputs.append(sent[0:i])
            targets.append(sent[i])
        y = to_categorical(targets, vocabSize)
        inputSeq = pad_sequences(inputs, maxlen=maxLen)
        yield(inputSeq,y)
#         print("inputs:", inputs)
#         print("-"*50)
#         print("targets:", targets)
#         print("-"*50)

# def cf():
#     for i in range(3):
#         yield i*i 
        
# obj = cf()
# for i in obj : 
#     print(i)
    
# for i , (x,y) in enumerate(genData(doc, maxLen, vocabSize)) :
#     print(i)
#     print("x", x.shape, "\n", x)
#     print("y", y.shape, "\n", y)

import numpy as np
xdata = [] 
ydata = [] 
for (x,y) in genData(doc,maxLen, vocabSize):
    xdata.append(x)
    ydata.append(y)
xdata = np.concatenate(xdata)
ydata = np.concatenate(ydata)

from keras.layers import Input, Embedding, Dense , LSTM ,Dropout
from keras.models import Sequential

# 모델 작성 
model = Sequential() 
model.add(Embedding(vocabSize,100 , input_length = maxLen)) #1165, 100 
model.add(LSTM(100, return_sequences = False ))
model.add(Dropout(0.5))
model.add(Dense(vocabSize, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer ="rmsprop", metrics = ["accuracy"])

model.fit(xdata,ydata, epochs = 500, batch_size=800)

model.save("tentGen.hdf5") # 모델 저장 

# --------------------------------------------------------------------------------------------------