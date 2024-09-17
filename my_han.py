import torch
import torch.nn as nn
from d2l import torch as d2l
from my_attention import MyAttention
from my_fc import MyFC


class MyHAN(nn.Module):
    def __init__(self, W_e, word_vec_size, gru_hidden_size):
        super(MyHAN, self).__init__()
        context_vec_size = 2*gru_hidden_size
        self.context_vec_size = context_vec_size
        self.W_e = W_e
        
        self.word_gru = nn.GRU(input_size=word_vec_size, hidden_size=gru_hidden_size, 
                               batch_first=True, device=d2l.try_gpu(), bidirectional=True)
        self.word_attention = MyAttention(gru_hidden_size*2, d2l.try_gpu())
        self.sente_gru = nn.GRU(input_size=context_vec_size, hidden_size=gru_hidden_size, 
                                batch_first=True, device=d2l.try_gpu(), bidirectional=True)
        self.sente_attention = MyAttention(gru_hidden_size*2, d2l.try_gpu())
        self.fc = MyFC(inputs_size=context_vec_size, outputs_size=1, device=d2l.try_gpu())
        
        
    def forward(self, doc):
        S = torch.empty((len(doc), self.context_vec_size))
        count = 0
        for sentence in doc:
            x_i = []
            for w_it in sentence:
                try: 
                    x_it = self.W_e.wv.get_vector(w_it).tolist()
                    x_i.append(x_it)
                except:
                    pass
            if(len(x_i) > 0):
                x_i = torch.Tensor(x_i)
                h_i, _ = self.word_gru(x_i)
                s_i = self.word_attention.forward(h_i)
                S[count] = s_i
                count += 1

        h_i, _ = self.sente_gru(S) # For sentence
        v = self.sente_attention.forward(h_i) # For sentence

        p = self.fc.forward(v);
        return p;