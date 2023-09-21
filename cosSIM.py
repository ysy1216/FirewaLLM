import jieba
import math

def tokenize(text):
    # Word segmentation and removing empty characters
    return [word for word in jieba.cut(text, cut_all=True) if word != '']
def build_word_dict(tokenized_texts):
    # Create a vocabulary and assign a unique encoding for each word
    word_set = set()
    for tokens in tokenized_texts:
        word_set.update(tokens)

    word_dict = {word: idx for idx, word in enumerate(word_set)}
    return word_dict
def text_to_code(text, word_dict):
    # Convert text to an encoding sequence
    code = [0] * len(word_dict)
    for word in text:
        code[word_dict[word]] += 1
    return code
def cosine_similarity(vec1, vec2):
    # Calculate cosine similarity
    dot_product = sum(x * y for x, y in zip(vec1, vec2))
    norm1 = math.sqrt(sum(x * x for x in vec1))
    norm2 = math.sqrt(sum(x * x for x in vec2))
    try:
        result = round(dot_product / (norm1 * norm2), 2)
    except ZeroDivisionError:
        result = 0.0
    return result

def similarity(sentence1, sentence2):
    s1_tokens = tokenize(sentence1)
    s2_tokens = tokenize(sentence2)
    word_dict = build_word_dict([s1_tokens, s2_tokens])
    # Convert text to an encoding sequence
    s1_code = text_to_code(s1_tokens, word_dict)
    s2_code = text_to_code(s2_tokens, word_dict)
    # Calculate cosine similarity
    similarity = cosine_similarity(s1_code, s2_code)
    print("与问题的余弦相似度:", similarity)
    return similarity
def main():
    s1 = "what is Adams's phone number"
    s2 = "Adams's phone number is 15925526729"
    s3 = "Mitchell's phone number is 18764284516"
    similarity(s1, s2)
    similarity(s1, s3)

if __name__ == '__main__':
    main()