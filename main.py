'''
Repredocution of
  Yang. 2016. Hierarchical Attention Networks for Document Classification. 
  
HAN indicates Hierarchical Attention Networks
'''
##########################################################################

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from gensim.models import Word2Vec
import math
from stanfordcorenlp import StanfordCoreNLP as nlp
from my_sentences import MySentences
import my_gru
import numpy



##########################################################################
#                  Word Attention & Sentence Attention                   #
##########################################################################

def get_params_attention(inputs_size, device):
    params_w = inputs_size
    
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01
    
    W_w = normal((params_w, params_w))
    b_w = torch.zeros(params_w, device=device)
    
    params = [W_w, b_w]
    for param in params:
        param.requires_grad_(True)
    return params


def attention(inputs, params):
    W_w, b_w = params
    
    u_w = []
    for x in inputs:
        u_it = torch.tanh((W_w @ x) + b_w)
        u_w.append(u_it)
    alpha = torch.softmax(u_w);
        
    s = torch.sum(torch.mul(alpha, x))    
    return s
   
# def get_params():
#     return [get_params_gru(), get_params_attention()]
    
##########################################################################
#                                  HAN                                   #
##########################################################################

def han(sentences, W_e, state, params):
    """   
    Structure:    
    1. inputs -> words -> embded into vector(x)
    2. (x) are feeded to bi-gru(), produce (h), for words
    3. (h) is attached with an attention, produce (s), refers to Eq.(5)-Eq.(7)
    4. (s) are feeded to bi-gru(), produce (h), for sentences
    5. (h) is attached with an attention, produce (v), refers to Eq.(8)-Eq.(10)
    6. (v) is used to calculate final classification (p)
    7. return (p)
    """
    # Retrieve all needed parameters
    params_gru_word_f, params_gru_word_b, params_word_att, params_gru_sec_f, params_gru_sec_b, params_sec_att  = params
    state_gru_word_f, state_gru_word_b, state_word_att, state_gru_sec_f, state_gru_sec_b, state_sec_att = state

    
    # Embed the words to vectors through an embedding matrix W_e
    X_it = []
    for sentence in sentences:
        for w_it in sentence:
            x_it = W_e @ w_it
            X_it.append(x_it)
    
    # Complete Word Encoder
    gru_word_forward = gru(X_it, state_gru_word_f, params_gru_word_f)
    gru_word_backward = gru(reversed(X_it), state_gru_word_b, params_gru_word_b)
    h_w = torch.cat(gru_word_forward, gru_word_backward)
    
    # Complete word attention
    s = attention(h_w, state_word_att, params_word_att)
    
    # Complete Sentence Encoder
    gru_sec_forward = gru(s, state_gru_sec_f, params_gru_sec_f)
    gru_sec_backward = gru(reversed(s), state_gru_sec_b, params_gru_sec_b)
    h_s = torch.cat(gru_sec_forward, gru_sec_backward)
    
    # Complete Sentence attention
    v = attention(h_s, state_sec_att, params_sec_att)
    
    #Complete document classification
    p = torch.softmax(v)
    
    return p
    

def han_loss():
    return 0

class HANModel:
    """先构建一个 HAN 模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn): 
        """Constructor

        Args:
            vocab_size (int): 词向量长度
            num_hiddens (int): 隐藏层神经元个数
            device (_type_): cpu/gpu, 由 torch 库决定
            get_params (_type_): 函数指针，用于生成参数矩阵的
            init_state (_type_): 函数指针
            forward_fn (_type_): 函数指针，RNN-Forward
        """
        #初始化RNNModelScratch类的属性
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device) 
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state): #python的默认方法call
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params) #这里的括号也没懂

    def begin_state(self, batch_size, device): #定义的方法begin_state
        return self.init_state(batch_size, self.num_hiddens, device) #一开始定义的batch_size

##########################################################################
#                          Predict and Train                             #
##########################################################################

def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]] 
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])   


#@save
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
    
def grad_clipping(net, theta):  #@save
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
            
            
# 这里也写了一个HANModel，类似RNNModel，用来创建一个类的对象
# net = HANModel
# 然后就可以使用 predict_ch8() 和 train_ch8() 函数进行训练了

##########################################################################
#                               Main part                                #
##########################################################################

word_vec_size = 200
gru_dim = 50
context_vec_size = 100
mini_batch = 64

data_folder = 'G:/NLP/word2vec/yelp_dataset_review/'
train_indices_dir = 'G:/NLP/word2vec/train_indices.pkl'
test_indices_dir = 'G:/NLP/word2vec/test_indices.pkl'
sentences = MySentences(data_folder, train_indices_dir, test_indices_dir, mini_batch)

# Get a batch of sentences
batch_sentences, max_sentence_len, is_senteces_end = sentences.get_a_train_batch()

# Embedding
W_e = Word2Vec.load('./word2vec/trained_model/my_word2vec_model_00')

# Word Encoder&Sentence Encoder(Using bidirectional GRU) 
gru_params_f = my_gru.get_params(word_vec_size, gru_dim, d2l.try_gpu())
gru_h_f = my_gru.init_state(mini_batch, gru_dim, d2l.try_gpu())
gru_params_b = my_gru.get_params(word_vec_size, gru_dim, d2l.try_gpu())
gru_h_b = my_gru.init_state(mini_batch, gru_dim, d2l.try_gpu())

for i in range(0, max_sentence_len):
    batch_wordvec_f = [] # Forward
    batch_wordvec_b = [] # Backward
    for j in range(0, mini_batch):
        # Get vector of words in forward
        vec = []       
        try:
            vec = W_e.wv.get_vector(batch_sentences[j][i]).tolist()
        except:
            vec = [0]*word_vec_size
        batch_wordvec_f.append(vec)
        # Get vector of words in forward
        vec = []
        try:
            vec = W_e.wv.get_vector(batch_sentences[j][max_sentence_len-1-i]).tolist()
        except:
            vec = [0]*word_vec_size
        batch_wordvec_b.append(vec)
    batch_wordvec_f = torch.Tensor(batch_wordvec_f)
    batch_wordvec_b = torch.Tensor(batch_wordvec_b)
    gru_h_f = my_gru.gru(batch_wordvec_f, gru_h_f, gru_params_f)
    gru_h_b = my_gru.gru(batch_wordvec_b, gru_h_b, gru_params_b)
# Bi-GRU = [forward, backward]
# gru_out = torch.cat(gru_h_f, gru_h_b)
    count = 0
    for X in gru_h_f:
        print(X)
        count += 1
        if count > 5:
            break;


