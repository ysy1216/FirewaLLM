# -*- coding: utf-8 -*-
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#Import corpus
def importdata(tag, flag):
    text_list = []
    if tag == "true":
        file_path = "data/Info/info2.txt"
        with open(file_path, encoding="UTF-8") as f:
            text = f.read()
            if flag:
                text_jieba = jieba.cut_for_search(text)
            else:
                text_jieba = jieba.lcut(text)
            text_list.append(" ".join(text_jieba))
    else:
        for i in range(3, 13):
            file_path = f"data/Info/info{i}.txt"
            with open(file_path, encoding="UTF-8") as f:
                text = f.read()
                text_jieba = jieba.lcut(text)
                text_list.append(" ".join(text_jieba))
    return text_list

#Import stop list and remove useless text
def stop_word():
    stopword_list = []
    for i in range(1, 5):
        file_path = f"data/StopWord/StopWord{i}.txt"
        with open(file_path, encoding="UTF-8") as f:
            stopwords = f.read().split("\n")
            stopword_list.extend(stopwords)
    return stopword_list

#Use jieba participle in sentences
def process_sentence(flag, sentence):
    if flag:
        sentence_jieba = jieba.lcut_for_search(sentence)
        sentence_jieba = " ".join(sentence_jieba)
    else:
        sentence_jieba = sentence
    return [sentence_jieba]

#Using Tfidf to Obtain Corpus Features
def vectorize_data(data, sentence_jieba):
    vectorizer = TfidfVectorizer(stop_words=stop_word())
    X = vectorizer.fit_transform(data).toarray()
    X_fea = vectorizer.get_feature_names_out()
    result = vectorizer.transform(sentence_jieba).toarray()
    X_pd = pd.DataFrame(result, columns=X_fea)
    return X_pd
#Compare the input sentences with the features of the corpus
def getFeature(tag, sentence, flag=True):
    data = importdata(tag, flag)
    st = process_sentence(flag, sentence)
    X_pd = vectorize_data(data, st)
    word_list = []
    X_pd_sort = X_pd.sort_values(by=0, axis=1, ascending=False)
    for i in range(0, 50):
        if X_pd_sort.iat[0, i] >= 0.20:
            tmp = X_pd_sort.iloc[:, i]
            tmp_frame = tmp.to_frame()
            word = "".join(tmp_frame.columns.tolist())
            if word not in word_list:
                word_list.append(word)
        else:
            break
    if len(word_list) == 0 and flag:
        word_list = getFeature(tag, sentence, False)
    return word_list
