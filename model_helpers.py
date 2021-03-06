from torchtext.data.utils import get_tokenizer
import spacy 
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
from nltk.corpus import words
import time 
import math 
# NLP model for tok2vec
config = {"model": DEFAULT_TOK2VEC_MODEL}
NLP = spacy.load("en_core_web_sm")
NLP.remove_pipe("tagger")
NLP.remove_pipe("senter")
NLP.remove_pipe("parser")
NLP.remove_pipe("lemmatizer")
NLP.remove_pipe("attribute_ruler")
NLP.remove_pipe("ner")


# Tokenizer for data preprocessing
corpus = set(words.words())
tokenizer = get_tokenizer("spacy", 'en')

def preprocess_text(s):
    new_s = s.lower().replace("-", "")
    news = tokenizer(new_s)
    res = ["<SOS>"]
    for token in news:
        if token != " ":
            if token == "\r\n" or token == "\r\n ":
                res.append("<EOL>")
            elif token in corpus:
                res.append(token)
    res.append("<EOS>")
    return res

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

