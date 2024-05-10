import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from wespeaker.models.ssl.WavLM_Large import *
from torch.nn.utils import remove_weight_norm
from wespeaker.models.ssl.modules import GradMultiply


class MHFA(nn.Module):
    def __init__(self,head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()
        self.weights_k = nn.Parameter(data=torch.ones(25),requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(25),requires_grad=True)
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb*self.cmp_dim, self.ous_dim)

    def forward(self,x):
        # X shape is [Batch, Dim, Frame_len, Nb_Layer]
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k,dim=-1)),dim=-1).transpose(1,2)
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v,dim=-1)),dim=-1).transpose(1,2)

        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        att_k = self.att_head(k)
        v = v.unsqueeze(-2)
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k,dim=1).unsqueeze(-1)),dim=1)
        b,h,f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b,-1)
        outs = self.pooling_fc(pooling_outs)
        return outs

class WavLM_Large_MHFA(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group=1, cnn_scale=1.0,layer_drop=0.05,frozen=False):
        super(WavLM_Large_MHFA, self).__init__()
        checkpoint = torch.load(model_path)
        checkpoint['cfg']['encoder_layerdrop']=layer_drop
        checkpoint['cfg']['feature_grad_mult']=cnn_scale
        cfg = WavLMConfig(checkpoint['cfg'])
        print('During the training, SSL is kept frozen:{}\n'.format(frozen))
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])
        self.frozen = frozen
        self.back_end = MHFA(inputs_dim=1024, head_nb=head_nb,outputs_dim=embed_dim)
        self.feature_grad_mult = 0.02

    def forward(self,wav_and_flag):
        
        x = wav_and_flag
        if self.frozen:
            with torch.no_grad():
                rep, layer_results = self.model.extract_features(x[:,:480000], output_layer=25)
        else:
            rep, layer_results = self.model.extract_features(x[:,:480000], output_layer=25)

        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
        x = GradMultiply.apply(x, self.feature_grad_mult)
        
        spk_embedding = self.back_end(x)
        
        return spk_embedding


    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            
            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);
