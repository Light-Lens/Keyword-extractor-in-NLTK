from nltk import tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from operator import itemgetter
import math

sentence = "Hello I'm Oxygen. I'm a natural language interface having access to all Horizon Security Networks, Defence Mechanism Systems and Srijan Srivastava's Private Server Protocols."

stop_words = set(stopwords.words('english'))
total_words = sentence.split()
total_word_length = len(total_words)

total_sentences = tokenize.sent_tokenize(sentence)
total_sent_len = len(total_sentences)

tf_score = {}
for each_word in total_words:
    each_word = each_word.replace('.','')
    if each_word not in stop_words:
        if each_word in tf_score:
            tf_score[each_word] += 1
        else:
            tf_score[each_word] = 1

tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())

def check_sent(word, sentences):
    final = [all([w in x for w in word]) for x in sentences]
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

idf_score = {}
for each_word in total_words:
    each_word = each_word.replace('.','')
    if each_word not in stop_words:
        if each_word in idf_score:
            idf_score[each_word] = check_sent(each_word, total_sentences)
        else:
            idf_score[each_word] = 1

idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())
tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

def get_top_n(dict_elem, n=None):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n])
    return result

print(list(get_top_n(tf_idf_score).keys()))
