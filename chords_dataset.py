from ast import literal_eval
from torchtext.data.utils import get_tokenizer
import numpy as np 
from torch.utils.data import Dataset
from torch import Tensor
from math import floor


class ChordsDataset(Dataset):
    def __init__(self, df, tok2vec):
        self.vectorizer = tok2vec
        self.lyrics = [literal_eval(str(c)) for c in df.lyrics]
        self.chords = [literal_eval(c) for c in df.chords] 
        self.chords_set = set([item for sublist in self.chords for item in sublist] + ["<SOS>", "<EOS>"])
        self.id2chord = {i:k for i,k in enumerate(self.chords_set)}
        self.word2vec = dict()
        self.chord2vec = dict()
        self.chord2id = {k:i for i,k in self.id2chord.items()}
    def __len__(self):
        return len(self.lyrics)
    def to_one_hot(self,chord):
        if chord in self.chord2vec:
            return self.chord2vec[self.id2chord[chord]]
        else:
            vec = np.zeros(len(self.chords_set))
            vec[chord] = 1
            self.chord2vec[self.id2chord[chord]] = vec
            return vec
    def to_chord_id(self,chord):
        return self.chord2id[chord]
        
    def get_train_test_valid_indexes(self, train_prop, test_prop, validation_prop):

        assert (train_prop + test_prop + validation_prop) == 1
        n = len(self.lyrics)
        l = np.array(range(n))
        np.random.shuffle(l)
        train_indexes = l[0:floor(n*train_prop)]
        test_indexes = l[floor(n*train_prop):(floor(n*train_prop) + floor(n*test_prop))]
        validation_indexes = l[(floor(n*train_prop) + floor(n*test_prop)):n]
        return train_indexes,test_indexes, validation_indexes

    def vectorize(self, word):
        if word in self.word2vec:
            return self.word2vec[word]
        vec = np.zeros(100)
        if word == "<SOS>": # Start of sequence token
            vec[96] = 1
            self.word2vec[word] = vec
        elif word == "<EOL>": # End of line token 
            vec[97] = 1
            self.word2vec[word] = vec
        elif word == "<EOS>": # End of sequence token
            vec[98] = 1
            self.word2vec[word] = vec
        else:
            vec = np.append(self.vectorizer(str(word)).vector, [0,0,0,0])
            self.word2vec[word] = vec
        return vec

    def decode_chord_tensor(self, tensor: Tensor) -> list:
        return [self.id2chord[int(idx.item())] for idx in tensor]
    
    def __getitem__(self,idx):

        return {
            "lyrics": Tensor([self.vectorize(word) for word in  self.lyrics[idx] ]),
            "chords": Tensor([self.to_chord_id(chord) for chord in (["<SOS>"] + self.chords[idx]+["<EOS>"])])
            }
