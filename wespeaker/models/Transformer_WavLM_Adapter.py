import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from wespeaker.models.ssl.WavLM import *
from torch.nn.utils import remove_weight_norm
from wespeaker.models.ssl.modules import GradMultiply
from wespeaker.models.ssl_backend import *
from wespeaker.models.ssl.adapter_wavlm.LoRa_WavLM import WavLM as LoRa_WavLM 
from wespeaker.models.ssl.adapter_wavlm.MAM_WavLM import WavLM as MAM_WavLM 
from wespeaker.models.ssl.adapter_wavlm.Parallel_WavLM import WavLM as Parallel_WavLM 
from wespeaker.models.ssl.adapter_wavlm.Prefix_WavLM import WavLM as Prefix_WavLM 
from wespeaker.models.ssl.adapter_wavlm.Seq_WavLM import WavLM as Seq_WavLM 

class WavLM_Base_Adapter(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group, adapter_type=None, adapter_dim=128, cnn_scale=0.0, layer_drop=0.05):
        super(WavLM_Base_Adapter, self).__init__()
        checkpoint = torch.load(model_path)
        checkpoint['cfg']['encoder_layerdrop']=layer_drop
        checkpoint['cfg']['feature_grad_mult']=cnn_scale
        print(adapter_type)        
        if adapter_type is not None:
            checkpoint['cfg']['adapter_dim'] = adapter_dim
            cfg = WavLMConfig(checkpoint['cfg'])
            if adapter_type == 'SeqAdapter':
                self.model = Seq_WavLM(cfg)
            elif adapter_type == 'ParallelAdapter':
                self.model = Parallel_WavLM(cfg)
            elif adapter_type == 'MAMAdapter':
                self.model = MAM_WavLM(cfg)
            elif adapter_type == 'LoRaAdapter':
                self.model = LoRa_WavLM(cfg)
            elif adapter_type == 'PrefixAdapter':
                self.model = Prefix_WavLM(cfg)                                                
        else:
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

        rep, layer_results = self.model.extract_features(x[:,:16000*20], output_layer=13)

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
