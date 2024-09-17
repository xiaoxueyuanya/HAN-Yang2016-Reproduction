'''
Repredocution of
  Yang. 2016. Hierarchical Attention Networks for Document Classification. 
  
HAN indicates Hierarchical Attention Networks
'''
##########################################################################

import torch
from gensim.models import Word2Vec
from my_documents import MyDocuments
from my_han import MyHAN
   

word_vec_size = 200
gru_hidden_size = 50
context_vec_size = 100
mini_batch = 64 
num_epochs = 10

data_folder = 'G:/NLP/word2vec/yelp_dataset_review/'
train_indices_dir = 'G:/NLP/word2vec/train_indices.pkl' # random file indics number for training
test_indices_dir = 'G:/NLP/word2vec/test_indices.pkl' # random file indics number for test
documents = MyDocuments(data_folder, train_indices_dir, test_indices_dir, mini_batch)

# Get a batch of sentences
# batch_docs, max_sentence_len, is_senteces_end = documents.get_a_train_batch_doc()

# Embedding
W_e = Word2Vec.load('./word2vec/trained_model/my_word2vec_model_00')

model = MyHAN(W_e, word_vec_size, gru_hidden_size)

# Optimizer
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    # pass data into the model
    while True:
        doc_sentences, label = documents.get_a_train_doc()
        if doc_sentences == []:
            break
        output = model.forward(doc_sentences)
        
        # Forward
        #outputs = model(inputs)
        loss = abs(output - label) 

        # Backward and optimize
        optimizer.zero_grad()  # 清零梯度
        loss.sum().backward()        # 计算梯度
        optimizer.step()       # 更新参数

    # Print loss
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 打印模型参数
# print("Model parameters:", list(model.parameters()))