# mark.py

import re
import jieba
import torch
from model import BertTextClassifier, BertLstmClassifier
from transformers import BertTokenizer, BertConfig, AutoTokenizer
import random
from feature import getFeature, stop_word

class MaskHandler:
    def __init__(self, model_path):
        # Initialize the local BERT model
        self.labels = ["Y", "N"]
        self.bert_config = BertConfig.from_pretrained('bert-base-chinese')
        self.model = BertLstmClassifier(self.bert_config, len(self.labels))
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # Initialize the cloud-based roberta model
        self.cloud_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext-large")

    def classic(self, query):
        sensitive_words = []
        for tmp_text in jieba.lcut(query):
            #print(tmp_text)
            token = self.tokenizer(tmp_text, add_special_tokens=True, padding='max_length', truncation=True,
                                   max_length=512)
            input_ids = torch.tensor([token['input_ids']], dtype=torch.long)
            attention_mask = torch.tensor([token['attention_mask']], dtype=torch.long)
            token_type_ids = torch.tensor([token['token_type_ids']], dtype=torch.long)
            predicted = self.model(input_ids, attention_mask, token_type_ids)
            pred_label = torch.argmax(predicted, dim=1)

        return sensitive_words

    #Desensitization algorithm
    def mask_sensitive_info(self, text, sensitive, level, tag):
        #print("脱敏等级第" + (str)(level+1) + "级")
        for word in sensitive:
            # text_jieba = jieba.lcut(text)
            len_word = len(word)
            if len_word < 8:
                tmp_level = 0
            else:
                tmp_level = level
            if tag == "true":
                length = len(word)
            else:
                length = (int)(len_word / 10 + tmp_level + 1)
            list = range(0, len(word))
            py = random.sample(list, length)
            tmp_word = word
            for count in range(0, length):
                pos = py[count]
                masked_sensitive = tmp_word[:pos] + '*' + tmp_word[pos + 1:]
                tmp_word = masked_sensitive
            text = re.sub(word, masked_sensitive, text, flags=re.IGNORECASE)
        return text

#Divide sentences into punctuation marks
def fun_splite(text):
    pattern = r',|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|：|！| |…|（|）,·'
    result_list = re.split(pattern, text.strip())
    return result_list

#Split the sentence with a stop word list
def fun_splitein(text):
    sentence_depart = jieba.lcut(text.strip())
    stopwords = stop_word()
    outstr = ""
    for word in sentence_depart:
        if word not in stopwords:
            if word != "\t":
                outstr += word
                # outstr += " "
    return outstr

#Using models to determine whether sentences are sensitive
def fun_isSen(maskHandler, text, tag):
    flag = False
    token = maskHandler.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
    input_ids = torch.tensor([token['input_ids']], dtype=torch.long)
    attention_mask = torch.tensor([token['attention_mask']], dtype=torch.long)
    token_type_ids = torch.tensor([token['token_type_ids']], dtype=torch.long)
    predicted = maskHandler.model(input_ids, attention_mask, token_type_ids)
    output = torch.softmax(predicted, dim=1)
    print(output)
    #output [:, 1]is an insensitive probability
    if tag == "false":
        if output[:, 0].item() > 0.5:
            flag = True
    return flag

#Invert the insensitive phrase returned by tfidf and return the sensitive phrase
def getSen(nosen, text):
    sen = []
    text_jieba = jieba.lcut(text)
    for word in text_jieba:
        if word not in nosen and len(word) > 1:
            if word not in sen:
                sen.append(word)
    return sen
#When inputting multiple sentences, determine whether each sentence is sensitive and then desensitize it
def fun_1(text, selected_sen_level, tag):
    maskHandler = MaskHandler("model/sen_model.pkl")  # Sensitive model
    text_splite = fun_splite(text)
    tmp = text
    for tmp_text in text_splite:
        text_stop = fun_splitein(tmp_text)
        sen_fea = []
        if fun_isSen(maskHandler, tmp_text, tag):
            if tag == "false":
                sen_fea = getSen(getFeature(tag, text_stop), text_stop)
            else:
                sen_fea = getFeature(tag, tmp_text, True)
        #print(sen_fea)
        res = maskHandler.mask_sensitive_info(tmp, sen_fea, selected_sen_level, tag)
        tmp = res
    print(tmp)
    return tmp

if __name__ == '__main__':
    str = "陈宇阳的手机号码是多少?"
    fun_1(str, 1, "false")

