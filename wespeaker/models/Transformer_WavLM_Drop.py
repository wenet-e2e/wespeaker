import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from wespeaker.models.ssl.WavLM import *
from torch.nn.utils import remove_weight_norm
from wespeaker.models.ssl.modules import GradMultiply
from wespeaker.models.ssl_backend import *

class WavLM_Base_Drop(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group, fixed=False):
        super(WavLM_Base_Drop, self).__init__()
        checkpoint = torch.load(model_path)
        print(checkpoint['cfg']['encoder_layerdrop'])
        self.fixed_SSL = fixed
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = WavLM(cfg)
        # self.model = remove_weight_norm(self.model)
        self.loadParameters(checkpoint['model'])
        if pooling == 'MHFA':
            self.back_end = MHFA(head_nb=head_nb,outputs_dim=embed_dim)
        elif pooling == 'G_MHFA':
            self.back_end = MHFA_Group(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'QKV':
            self.back_end = MHFA_Dotproduct(compression_dim=256, outputs_dim=embed_dim)
        elif pooling == 'G_MHFA_MQSKMV':
            self.back_end = MHFA_Group_MQ_SK_MV(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'G_MHFA_MQMKSV':
            self.back_end = MHFA_Group_MQ_MK_SV(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'G_MHFA_Conv2D':
            self.back_end = MHFA_Group_Conv2D(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'MHFA_Context':
            self.back_end = MHFA_context(head_nb=head_nb,outputs_dim=embed_dim)
        elif pooling == 'G_MHFA_Conv2D_MeanStd':
            self.back_end = MHFA_Group_Conv2D_MeanStd(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'TSTP':
            self.back_end = TSTP(outputs_dim=embed_dim)
        elif pooling == 'ASTP':
            self.back_end = ASTP(outputs_dim=embed_dim)
        elif pooling == 'Last_ASTP':
            self.back_end = Last_ASTP(outputs_dim=embed_dim)
        elif pooling == 'CorrelationPoolingDrop':
            self.back_end = CorrelationPoolingDrop(outputs_dim=embed_dim)
        elif pooling == 'CorrelationPooling':
            self.back_end = CorrelationPooling(outputs_dim=embed_dim)        
        self.feature_grad_mult = 0.08

    def forward(self,wav_and_flag):
        
        x = wav_and_flag
        with torch.no_grad():
            rep, layer_results = self.model.extract_features(x[:,:16000*20], output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
        
        # x = GradMultiply.apply(x, self.feature_grad_mult)
        
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
