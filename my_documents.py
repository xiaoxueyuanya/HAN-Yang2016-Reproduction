import pickle
import json
import os
import re
import nltk

class MyDocuments(object):
    def __init__(self, dirname, train_indices_dir, test_indices_dir, 
                 mini_batch):
        self.dirname = dirname
        self.filenames = os.listdir(dirname) # [] get the filenames of the specified directory
        with open(train_indices_dir, 'rb') as file:
            self.train_indices = pickle.load(file)
        with open(test_indices_dir, 'rb') as file:
            self.test_indices = pickle.load(file)
        self.mini_batch = mini_batch
        self.train_file_count = 0
        filename = self.filenames[self.train_indices[self.train_file_count]]
        self.train_file = open(os.path.join(self.dirname, filename), 'r')
        print("Load documents in " + filename)
        self.test_file_count = 0
        self.test_file = open(os.path.join(
            self.dirname, self.filenames[self.test_indices[self.test_file_count]]), 'r')  
  
    def get_a_train_doc(self):
        doc_sentences = [] # tokenized
        label = []
        try:
            line = self.train_file.readline()
            data = json.loads(line) # a line is a dictionary of a review
            text = data["text"] # get the content of review
            label = data["stars"] # get the stars of the review
            sentences = nltk.sent_tokenize(text) # split into sentence
            for sentence in sentences:
                sente = re.sub('[^A-Za-z]+', ' ', sentence).strip().lower().split()
                if len(sente) > 0:
                    doc_sentences.append(sente)
        except:
            self.train_file_count += 1
            if self.train_file_count >= len(self.filenames):
                doc_sentences = []
                label = 0
            else:
                filename = self.filenames[self.train_indices[self.train_file_count]]
                self.train_file = open(os.path.join(self.dirname, filename), 'r')
                print("Load documents in " + filename)
                # put the tokenized sentences into doc_sentences
                doc_sentences, label = self.get_a_train_doc() 
        return doc_sentences, label
            
    def get_a_train_batch_doc(self): 
        batch_sentences = []
        batch_labels = []
        is_end_file = False
        for _ in range(0, self.mini_batch):
            doc_sentences, label = self.get_a_train_doc() 
            if len(doc_sentences) == 0: # for the last sentence
                batch_sentences.append([""])
                batch_labels.append(label)
                is_end_file = True
            else:
                batch_sentences.append(doc_sentences) # append sentences into the batch
                batch_labels.append(label)
        return batch_sentences, batch_labels, is_end_file