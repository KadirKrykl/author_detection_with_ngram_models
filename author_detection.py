import io, os
import re as re
import zipfile as zipfile
import sys
import random
import math 
import time
from functools import reduce

mytextzip = ''
docList={}
# {author:{doc_id:text}}
idx_ID=1
author = 0
docList[str(author)] = {}
with zipfile.ZipFile('30Columnists.zip') as z:
    for zipinfo in z.infolist():
        mytextzip = ''
        if zipinfo.filename.endswith('.txt') and re.search('raw_texts', zipinfo.filename):
            with z.open(zipinfo) as f:
                textfile = io.TextIOWrapper(f, encoding='cp1254', newline='')
                for line in textfile:
                    if len(line.strip()): mytextzip += ' ' + line.strip()
                document={
                    'text' : mytextzip
                }
                docList[str(author)][str(idx_ID)] = document
                if idx_ID % 50 == 0:
                    author+=1
                    if author != 30:
                        docList[str(author)] = {}
                idx_ID+=1
          

def testDataSplit(wholeData,testCount):
    # testCount = test count for each class
    test_y = []
    test_x = []
    for author , docs in wholeData.items():
        docCount = len(docs)
        for i in range(testCount):
            randomID = str(random.randint(int(list(docs.keys())[0]),int(list(docs.keys())[-1])))
            test_y.append(author)
            test_x.append(docs[randomID]['text'])
    return test_x, test_y

testX, testY = testDataSplit(docList,5)

# TOKENIZATION

# Non-breaking to normal space
NON_BREAKING = re.compile(u"\s+"), " "
# Multiple dot
MULTIPLE_DOT = re.compile(u"\.+"), "."
# Sentence dots
SENTENCE_DOT = re.compile(u"(?!\B\"\s+[^\"]*)[\.?!](?![^\"]*\s+\"\B)"), r" -d- "
# Merge multiple spaces.
ONE_SPACE = re.compile(r' {2,}'), ' '
# Numbers
NUMBERS= re.compile(r'[0-9]*[0-9]'), ' ' 
# 2.5 -> 2.5 - asd. -> asd . 
DOT_WITHOUT_FLOAT = re.compile("((?<![0-9])[\.])"), r' '
# 2,5 -> 2,5 - asd, -> asd , 
COMMA_WITHOUT_FLOAT = re.compile("((?<![0-9])[,])"), r' '
# doesn't -> doesn't  -  'Something' -> ' Something '
QUOTE_FOR_NOT_S = re.compile("\b(?<![n])[\'](?![t])\b"), r' '
AFTER_QUOTE_SINGLE_S = re.compile("\s+[s]\s+"), r' '
# Extra punctuations "!()
NORMALIZE = re.compile("([\–])"), r'-'
EXTRAS_PUNK = re.compile("([^\'\.\,\w\s\-\–])"), r' '

REGEXES = [
    NON_BREAKING,
    MULTIPLE_DOT,
    NUMBERS,
    QUOTE_FOR_NOT_S,
    AFTER_QUOTE_SINGLE_S,
    SENTENCE_DOT,
    DOT_WITHOUT_FLOAT,
    COMMA_WITHOUT_FLOAT,
    NORMALIZE,
    EXTRAS_PUNK,
    ONE_SPACE
]

def normalize(text):
    text = text.lower()
    for regexp, subsitution in REGEXES:
        text = regexp.sub(subsitution, text)
    return text

def tokenizer(text):
    normalizedText = normalize(text)
    tokens = normalizedText.split('-d-')
    tokens = "</s> <s>".join(tokens)
    tokens = tokens.split(' ')
    tokens[0] = "<s>"
    tokens[len(tokens)-1] = "</s>"
    return tokens


start_time = time.time()

for author, docs in docList.items():
    docList[author]['all_tokens'] = []
    for key,value in docs.items():
        if key != 'all_tokens':
            tokens = tokenizer(value['text'])
            docList[author][key]['tokens'] = tokens
            docList[author]['all_tokens'].extend(tokens)
    
elapsed_time = time.time() - start_time
print("Tokenize: "+str(elapsed_time))


# Language Models

def reducer(first, last):
    for item in last: 
        for item in last:
            first[item] = first.get(item, 0) + last.get(item, 0)
    return first



# 1-GRAM MODEL
start_time = time.time()

for author, docs in docList.items():
    a = docList[author]['all_tokens']
    tokensMap = map(lambda char: dict([[char, 1]]), a)
    wordFreq = reduce(reducer, tokensMap)
    totalWord = sum(list(wordFreq.values()))
    docList[author]['dl'] = totalWord
    docList[author]['TF'] = wordFreq
    docList[author]['1-gram'] = dict()
    docList[author]['1-gram'].update((k, v/totalWord) for k,v in docList[author]['TF'].items())
    
        
elapsed_time = time.time() - start_time
print("1-gram: "+str(elapsed_time))

# 2-GRAM MODEL
start_time = time.time()

for author, docs in docList.items():
    bifreq = map(lambda char: dict([[(docs['all_tokens'][char], docs['all_tokens'][char-1]),1]]), range(len(docs['all_tokens'])))
    bifreq = reduce(reducer, bifreq)
    docList[author]['2TF'] = bifreq
    docList[author]['2-gram'] = dict()
    docList[author]['2-gram'].update((k, v/docList[author]['TF'][k[1]]) for k,v in docList[author]['2TF'].items())

elapsed_time = time.time() - start_time
print("2-gram: "+str(elapsed_time))

# 3-GRAM MODEL
start_time = time.time()

for author, docs in docList.items():
    trigram = map(lambda char: dict([[(docs['all_tokens'][char], (docs['all_tokens'][char-1], docs['all_tokens'][char-2])),1]]), range(len(docs['all_tokens'])))
    trigram = reduce(reducer, trigram)
    trigram.update((k, v/docList[author]['2TF'][k[1]]) for k,v in trigram.items())
    docList[author]['3-gram'] = trigram

elapsed_time = time.time() - start_time
print("3-gram: "+str(elapsed_time))

"""
f = open("newFile.txt",'a')
for key,value in docList['0']['1-gram'].items():
    f.write("{0} => {1} \n".format(key,value))    
f.close()
"""

def classifier1gram(doc):
    tokens = tokenizer(doc)
    probs = [1 for i in docList.keys()]
    probs2 = [1 for i in docList.keys()]
    for author , properties in docList.items():
        sentenceProbs = [1]
        for token in tokens:
            if token in properties['1-gram'].keys():
                sentenceProbs[-1] *= properties['1-gram'][token]
            if token == '</s>':
                sentenceProbs.append(1)
        probs2[int(author)] = sum(sentenceProbs)
        probs[int(author)] = reduce((lambda x, y: x * y), sentenceProbs) 
    maxVal = max(probs2)
    return probs2.index(maxVal)

def classifier2gram(doc):
    tokens = tokenizer(doc)
    probs = [1 for i in docList.keys()]
    probs2 = [1 for i in docList.keys()]
    boundaries = list(map(lambda char: tuple([tokens[char], tokens[char-1]]), range(len(tokens))))
    for author , properties in docList.items():
        sentenceProbs = [1]
        for boundary in boundaries:
            if boundary in properties['2-gram'].keys():
                sentenceProbs[-1] *= properties['2-gram'][boundary]
            if boundary[0] == '</s>':
                sentenceProbs.append(1)
        probs2[int(author)] = sum(sentenceProbs)
        probs[int(author)] = reduce((lambda x, y: x * y), sentenceProbs) 
    maxVal = max(probs)
    return probs.index(maxVal)

def classifier3gram(doc):
    tokens = tokenizer(doc)
    probs = [1 for i in docList.keys()]
    probs2 = [1 for i in docList.keys()]
    boundaries = list(map(lambda char: tuple([tokens[char], (tokens[char-1],tokens[char-2])]), range(len(tokens))))
    for author , properties in docList.items():
        sentenceProbs = [1]
        for boundary in boundaries:
            if boundary in properties['3-gram'].keys():
                sentenceProbs[-1] *= properties['3-gram'][boundary]
            if boundary[0] == '</s>':
                sentenceProbs.append(1)
        probs2[int(author)] = sum(sentenceProbs)
        probs[int(author)] = reduce((lambda x, y: x * y), sentenceProbs) 
    
    maxVal = max(probs)
    return probs.index(maxVal)
    

a = classifier1gram(testX[0])
b = classifier2gram(testX[0])
c = classifier3gram(testX[0])
print(a,b,c)