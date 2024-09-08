import pickle
import json
import os
import re

class MySentences(object):
    def __init__(self, dirname, train_indices_dir, test_indices_dir, 
                 mini_batch):
        self.dirname = dirname
        self.filenames = os.listdir(dirname)
        with open(train_indices_dir, 'rb') as file:
            self.train_indices = pickle.load(file)
        with open(test_indices_dir, 'rb') as file:
            self.test_indices = pickle.load(file)
        self.mini_batch = mini_batch
        self.train_file_count = 0
        self.train_file = open(os.path.join(
            self.dirname, self.filenames[self.train_indices[self.train_file_count]]), 'r')
        self.test_file_count = 0
        self.test_file = open(os.path.join(
            self.dirname, self.filenames[self.test_indices[self.test_file_count]]), 'r')
  
    def get_a_train_sentence(self):
        sentence = []
        try:
            line = self.train_file.readline()
            data = json.loads(line)
            text = data["text"]
            sentence = re.sub('[^A-Za-z]+', ' ', text).strip().lower().split()
        except:
            self.train_file_count += 1
            if self.train_file_count >= len(self.filenames):
                sentence = []
            else:
                self.train_file = open(os.path.join(
                    self.dirname, self.filenames[self.train_indices[self.train_file_count]]), 'r')
                sentence = self.get_a_train_sentence()
        return sentence
            
    def get_a_train_batch(self): 
        batch_sentences = []
        max_sentece_len = 0
        is_end_file = False
        for i in range(0, self.mini_batch):
            sentence = self.get_a_train_sentence()
            if len(sentence) == 0:
                batch_sentences.append([""])
                is_end_file = True
            else:
                batch_sentences.append(sentence)
                if(len(sentence) > max_sentece_len):
                    max_sentece_len = len(sentence)
        return batch_sentences, max_sentece_len, is_end_file