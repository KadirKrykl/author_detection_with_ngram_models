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
author=0
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

def pre_porcess_tokenize_sentence(sentence):
    sentence = sentence.lower()
    for regexp, subsitution in REGEXES:
        sentence = regexp.sub(subsitution, sentence)
    return sentence


start_time = time.time()

for author, docs in docList.items():
    docList[author]['all_tokens'] = []
    for key,value in docs.items():
        if key != 'all_tokens':
            tokenizedText = pre_porcess_tokenize_sentence(value['text'])
            tokens = tokenizedText.split('-d-')
            tokens = "</s> <s>".join(tokens)
            tokens = tokens.split(' ')
            tokens[0] = "<s>"
            tokens[len(tokens)-1] = "</s>"
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
    wordBoundaries = map(lambda char: dict([[(docs['all_tokens'][char], docs['all_tokens'][char-1]),1]]), range(len(docs['all_tokens'])))
    wordBoundaries = reduce(reducer, wordBoundaries)
    wordBoundaries.update((k, v/docList[author]['TF'][k[1]]) for k,v in wordBoundaries.items())
    docList[author]['2-gram'] = wordBoundaries

elapsed_time = time.time() - start_time
print("2-gram: "+str(elapsed_time))

