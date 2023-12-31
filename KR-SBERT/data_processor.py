import csv

tags = ['CLA', 'ETC', 'FUT', 'GLO', 'LIT', 'SCI', 'SOC', 'SOF', 'VIR']

for tag in tags:
    f = open('data/' + tag + '_dict.tsv','r', encoding='utf-8') # *.dict.tsv 파일은 강의 인덱스 - 강의 이름+교수명 매핑 파일
    rdr = csv.reader(f, delimiter='\t')
    mapper = []
    for row in rdr:
        mapper.append(row[1])
    f.close()

    f = open('data/' + tag + '_data.tsv','r', encoding='utf-8') # *.data.tsv 파일은 강의평, 강의 인덱스 데이터 파일
    rdr = csv.reader(f, delimiter='\t')
    data = []
    for row in rdr:
        data.append(row)
    f.close()

    import random

    train_examples = []
    test_examples = []
    train_count = 8000 # hyper_params.
    test_count = 2000
    batch_size = 16
    epochs = 32

    preTrain = open('train/' + tag + '_preTrain_data_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    preTest = open('test/' + tag + '_predata_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    for k in list(range(len(data))):
        if(k % 5 != 0):
            preTrain.write(data[k][0].replace(',','') + ',' + data[k][1] + '\n')
        else:
            preTest.write(data[k][0].replace(',','') + ',' + data[k][1] + '\n')
    preTrain.close()
    preTest.close()

    preTrain = open('train/' + tag + '_preTrain_data_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    preTrainRdr = csv.reader(preTrain)
    preTest = open('test/' + tag + '_predata_' + str(train_count+test_count) + '.csv','r', encoding='utf-8')
    preTestRdr = csv.reader(preTest)

    train_data = [[] for _ in list(range(len(mapper)))]
    test_data = []
    for row in preTrainRdr:
        train_data[int(row[1])].append(str(row[0]))
    for row in preTestRdr:
        test_data.append(row)

    preTrain.close()
    preTest.close()

    for k in range(train_count): # training data 생성. (강의A_강의평1, 강의A_강의평2, 강의B_강의평1) 형식의 튜플을 train_example에 저장
        i = random.randint(0, len(train_data)-1)
        j = -1
        while(j <0 or j == i):
            j = random.randint(0, len(train_data)-1)

        train_examples.append([train_data[i][random.randint(0, len(train_data[i])-1)], train_data[i][random.randint(0, len(train_data[i])-1)], train_data[j][random.randint(0, len(train_data[j])-1)]])

    randomList = []
    k = 0
    while(k < test_count):
        l = random.randint(0, len(data)-1)

        if(l in randomList):
            continue
        else:
            test_examples.append(data[l])
            k = k+1

    f = open('train/' + tag + '_train_data_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    for train_example in train_examples:
        f.write(train_example[0].replace(',','') + ',' + train_example[1].replace(',','') + ',' + train_example[2].replace(',','') + '\n')
    f.close()

    f = open('test/' + tag + '_test_data_' + str(train_count+test_count) + '.csv','w', encoding='utf-8')
    for test_example in test_examples:
        f.write(test_example[0].replace(',','') + ',' + test_example[1] + '\n')
    f.close()
