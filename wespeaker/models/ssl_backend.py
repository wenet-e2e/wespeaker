import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

class MHFA(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        # B x C x T x L -> B x C x T -> B x T x C
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2) 

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k) # B x T x C -> B x T x H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B, T, 1

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        # pooling_outs dim: B x H x C
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1) # B x H x C -> B x (H x C)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs

class MHFA_context(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA_context, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim*3, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k) # B, T, F
        v = self.cmp_linear_v(v)

        context_mean = torch.mean(k, dim=1, keepdim=True).expand_as(k)
        context_std = torch.sqrt(torch.var(k, dim=1, keepdim=True) + 1e-7).expand_as(k)
        k_in = torch.cat((k, context_mean, context_std), dim=-1)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k_in) # B, T, H

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B, T, 1

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs



class MHFA_L(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA_L, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(25), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(25), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs


class MHFA_Dotproduct(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=256, outputs_dim=256):
        super(MHFA_Dotproduct, self).__init__()

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define learnable weights for key, value, and query computations across layers
        self.weights_k = nn.Parameter(torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(torch.ones(13), requires_grad=True)
        self.weights_q = nn.Parameter(torch.ones(13), requires_grad=True)

        # Define compression linear layers for keys, values, and queries
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_q = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.cmp_dim * 2, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute key, value, and query by taking a weighted sum of input across layers
        k = torch.sum(x.mul(F.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        v = torch.sum(x.mul(F.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)
        q = torch.sum(x.mul(F.softmax(self.weights_q, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys, values, and queries through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)
        q = self.cmp_linear_q(q)

        # Reshape q, k, v for multi-head attention
        batch_size = q.size(0)
        q = q.view(batch_size, -1, self.head_nb, self.cmp_dim // self.head_nb).transpose(1, 2)
        k = k.view(batch_size, -1, self.head_nb, self.cmp_dim // self.head_nb).transpose(1, 2)
        v = v.view(batch_size, -1, self.head_nb, self.cmp_dim // self.head_nb).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.cmp_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v)

        # Reshape attention output to original size
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.head_nb * (self.cmp_dim // self.head_nb))

        mean = attn_output.mean(dim=1)
        std = attn_output.std(dim=1)
        combined = torch.cat((mean, std), dim=1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(combined)


        return outs

class MHFA_ChannelAtt(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=256, outputs_dim=256):
        super(MHFA_ChannelAtt, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)

        self.head_attention = nn.Linear(2 * self.cmp_dim, head_nb)
        
        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(2 * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        # B, F_len, Dim
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        # B, F_len, Dim
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        # Compute attention weights using compressed keys
        att_k = self.att_head(k)

        # Adjust dimensions for computing attention output
        v = v.unsqueeze(-2) # B, F_len, 1, CDim
        v_squared = v ** 2
        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        alpha = nn.functional.softmax(att_k, dim=1).unsqueeze(-1) # B, F_len, Head, 1
        mean = torch.sum(v.mul(alpha), dim=1) # B, Head, CDim
        var = torch.sum(v_squared.mul(alpha), dim=1) - mean**2

        std = torch.sqrt(var.clamp(min=1e-7)) # B, Head, CDim
        out = torch.cat((mean, std), dim=2) # B, Head, 2 * CDim

        head_att_weights = F.softmax(self.head_attention(out.mean(dim=1)), dim=1)  # B, Head
        head_att_weights = head_att_weights.unsqueeze(-1)  # B, Head, 1
        
        out = torch.sum(out * head_att_weights, dim=1)  # B, 2 * CDim

        # Pass through fully connected layer to get the final output
        out = self.pooling_fc(out)

        return out

class MHFA_Group_Conv2D(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256,group_nb=2):
        super(MHFA_Group_Conv2D, self).__init__()
        # Multi Q + Single K + Single V
        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.group_nb = group_nb

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        # group_len
        group_len = self.head_nb // self.group_nb
        self.att_head = nn.Conv2d(1,self.group_nb,(group_len+1,self.cmp_dim),bias=False,stride=1,padding=(group_len//2, 0)) # Kernel Size [G_len, F]
        # self.att_head = []
        # self.att_head = nn.ModuleList([nn.Linear(self.cmp_dim, self.head_nb // self.group_nb) for i in range(self.group_nb)])

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.group_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        batch_size = k.size(0)
        k = self.cmp_linear_k(k) # B, T, F
        # k = k.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) # Groups, B, T, D
        v = self.cmp_linear_v(v) #B, T, F
        # v = v.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) 

        k_att = self.att_head(k.unsqueeze(1)) # B, Head, T, 1
        k_att = k_att.permute(0,2,1,3) #  # B, F_len, 1, Head
        # print(k_att.shape)

        v = v.unsqueeze(-2) # B, T, 1, D

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(k_att, dim=1)), dim=1)

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)
        return outs

class MHFA_Group(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256,group_nb=2):
        super(MHFA_Group, self).__init__()
        # Multi Q + Multi K + Multi V
        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.group_nb = group_nb

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim*self.group_nb)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim*self.group_nb)

        # Define linear layer to compute multi-head attention weights
        self.att_head = []
        self.att_head = nn.ModuleList([nn.Linear(self.cmp_dim, self.head_nb // self.group_nb) for i in range(self.group_nb)])

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        batch_size = k.size(0)
        k = self.cmp_linear_k(k)
        k = k.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) # Groups, B, T, D
        v = self.cmp_linear_v(v)
        v = v.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) 

        group_embeddings = []
        for i in range(self.group_nb):
            att_k = self.att_head[i](k[i])
            v_sub = v[i].unsqueeze(-2)
            mean = torch.sum(v_sub.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)
            mean = mean.contiguous().view(batch_size,-1)
            group_embeddings.append(mean)
        
        group_embeddings_out = torch.cat(group_embeddings,dim=1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(group_embeddings_out)

        return outs


class MHFA_Group_MQ_MK_SV(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256,group_nb=2):
        super(MHFA_Group_MQ_MK_SV, self).__init__()
        # Multi Q + Multi K + Single V
        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.group_nb = group_nb

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim * self.group_nb)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim * 1)

        # Define linear layer to compute multi-head attention weights
        self.att_head = []
        self.att_head = nn.ModuleList([nn.Linear(self.cmp_dim, self.head_nb // self.group_nb) for i in range(self.group_nb)])

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        batch_size = k.size(0)
        k = self.cmp_linear_k(k)
        k = k.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) # Groups, B, T, D
        v = self.cmp_linear_v(v)
        v = v.view(batch_size, -1, 1, self.cmp_dim).permute(2,0,1,3) 

        group_embeddings = []
        for i in range(self.group_nb):
            att_k = self.att_head[i](k[i])
            v_sub = v[0].unsqueeze(-2)
            mean = torch.sum(v_sub.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)
            mean = mean.contiguous().view(batch_size,-1)
            group_embeddings.append(mean)
        
        group_embeddings_out = torch.cat(group_embeddings,dim=1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(group_embeddings_out)

        return outs


class MHFA_Group_MQ_SK_MV(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256,group_nb=2):
        super(MHFA_Group_MQ_SK_MV, self).__init__()
        # Multi Q + Multi K + Single V
        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.group_nb = group_nb

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim * 1)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim * self.group_nb)

        # Define linear layer to compute multi-head attention weights
        self.att_head = []
        self.att_head = nn.ModuleList([nn.Linear(self.cmp_dim, self.head_nb // self.group_nb) for i in range(self.group_nb)])

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.head_nb * self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        batch_size = k.size(0)
        k = self.cmp_linear_k(k)
        k = k.view(batch_size, -1, 1, self.cmp_dim).permute(2,0,1,3) # Groups, B, T, D
        v = self.cmp_linear_v(v)
        v = v.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) 

        group_embeddings = []
        for i in range(self.group_nb):
            att_k = self.att_head[i](k[0])
            v_sub = v[i].unsqueeze(-2)
            mean = torch.sum(v_sub.mul(nn.functional.softmax(att_k, dim=1).unsqueeze(-1)), dim=1)
            mean = mean.contiguous().view(batch_size,-1)
            group_embeddings.append(mean)
        
        group_embeddings_out = torch.cat(group_embeddings,dim=1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(group_embeddings_out)

        return outs


class MHFA_Group_Conv2D_MeanStd(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256,group_nb=2):
        super(MHFA_Group_Conv2D_MeanStd, self).__init__()
        # Multi Q + Single K + Single V
        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.group_nb = group_nb

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)

        # Define linear layer to compute multi-head attention weights
        # group_len
        group_len = self.head_nb // self.group_nb
        self.att_head = nn.Conv2d(1,self.group_nb,(group_len+1,self.cmp_dim),bias=False,stride=1,padding=(group_len//2, 0)) # Kernel Size [G_len, F]
        # self.att_head = []
        # self.att_head = nn.ModuleList([nn.Linear(self.cmp_dim, self.head_nb // self.group_nb) for i in range(self.group_nb)])

        # Define a fully connected layer for final output
        self.pooling_fc = nn.Linear(self.group_nb * self.cmp_dim * 2, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)

        # Compute the value in a similar fashion
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v, dim=-1)), dim=-1).transpose(1, 2)

        # Pass the keys and values through compression linear layers
        batch_size = k.size(0)
        k = self.cmp_linear_k(k) # B, T, F
        # k = k.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) # Groups, B, T, D
        v = self.cmp_linear_v(v) #B, T, F
        # v = v.view(batch_size, -1, self.group_nb, self.cmp_dim).permute(2,0,1,3) 

        k_att = self.att_head(k.unsqueeze(1)) # B, Head, T, 1
        k_att = k_att.permute(0,2,1,3) #  # B, F_len, 1, Head
        # print(k_att.shape)

        v = v.unsqueeze(-2) # B, T, 1, D
        v2 = v**2
        alpha = nn.functional.softmax(k_att, dim=1)

        # Compute attention output by taking weighted sum of values using softmaxed attention weights
        pooling_outs = torch.sum(v.mul(alpha), dim=1)
        pooling_outs_var = torch.sum(v2.mul(alpha), dim=1) - pooling_outs**2
        std = torch.sqrt(pooling_outs_var.clamp(min=1e-7))

        # Reshape the tensor before passing through the fully connected layer
        b, h, f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b, -1)
        std = std.reshape(b,-1)

        outs = torch.cat([pooling_outs, std],dim=1)
        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(outs)

        return outs

class LWAP_Mean(nn.Module):
    def __init__(self, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(LWAP_Mean, self).__init__()

        # Define learnable weights for computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)

        self.pooling_fc = nn.Linear(self.cmp_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim (Feature), Frame_len (Time), Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        x = self.cmp_linear_k(k)
        x = x.transpose(1,2)

        pooling_outs = torch.mean(x, dim=-1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs
class LWAP_PoolDim(nn.Module):
    def __init__(self, pool_dim=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(LWAP_PoolDim, self).__init__()

        # Define learnable weights for computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.pol_dim = pool_dim
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)

        self.pooling_fc = nn.Linear(self.cmp_dim*self.pol_dim, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim (Feature), Frame_len (Time), Nb_Layer]
        batch_size, _, frame_len, _ = x.shape

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        x = self.cmp_linear_k(k)
        x = x.transpose(1,2)
        
        # Average pooling - frame_len must be divisible by pool_dim
        avg_pooling = nn.AvgPool1d(frame_len//self.pol_dim, stride=None)
        pooling_outs = avg_pooling(x)

        pooling_outs = pooling_outs.reshape(batch_size, -1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(pooling_outs)

        return outs

class TSTP(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(TSTP, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.pooling_fc = nn.Linear(self.cmp_dim*2, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        x = self.cmp_linear_k(k)
        x = x.transpose(1,2)
        pooling_mean = x.mean(dim=-1)
        pooling_std = torch.sqrt(torch.var(x, dim=-1) + 1e-7)
        pooling_mean = pooling_mean.flatten(start_dim=1)
        pooling_std = pooling_std.flatten(start_dim=1)
        stats = torch.cat((pooling_mean, pooling_std), 1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(stats)

        return outs


class ASTP(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(ASTP, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.linear1 = nn.Conv1d(3*self.cmp_dim, 256,kernel_size=1)
        self.linear2 = nn.Conv1d(256, self.cmp_dim,kernel_size=1)

        self.pooling_fc = nn.Linear(self.cmp_dim*2, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        x = self.cmp_linear_k(k)
        x = x.transpose(1,2)

        context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
        context_std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
        x_in = torch.cat((x, context_mean, context_std), dim=1)
        # print(x_in.shape)

        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        stats = torch.cat([mean, std], dim=1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(stats)

        return outs

class Last_ASTP(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(Last_ASTP, self).__init__()

        # Define learnable weights for key and value computations across layers
        # self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.linear1 = nn.Conv1d(3*self.cmp_dim, 256,kernel_size=1)
        self.linear2 = nn.Conv1d(256, self.cmp_dim,kernel_size=1)

        self.pooling_fc = nn.Linear(self.cmp_dim*2, self.ous_dim)

    def forward(self, x):
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        # k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        k = x[:,:,:,-1].transpose(1, 2)
        x = self.cmp_linear_k(k)
        x = x.transpose(1,2)

        context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
        context_std = torch.sqrt(
            torch.var(x, dim=-1, keepdim=True) + 1e-7).expand_as(x)
        x_in = torch.cat((x, context_mean, context_std), dim=1)
        # print(x_in.shape)

        alpha = torch.tanh(
            self.linear1(x_in))  # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x**2), dim=2) - mean**2
        std = torch.sqrt(var.clamp(min=1e-7))
        stats = torch.cat([mean, std], dim=1)

        # Pass through fully connected layer to get the final output
        outs = self.pooling_fc(stats)

        return outs

class CorrelationPoolingDrop(nn.Module):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(CorrelationPoolingDrop, self).__init__()

        # Define learnable weights for key and value computations across layers
        self.weights_k = nn.Parameter(data=torch.ones(13), requires_grad=True)

        # Initialize given parameters
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim

        p_drop=0.25
        self.drop_f = p_drop != 0
        if self.drop_f:
            self.dropout = nn.Dropout2d(p=p_drop)

        # Define compression linear layers for keys and values
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.pooling_fc = nn.Linear(int(128*(128-1)/2), self.ous_dim)
    def forward(self, x):
        # print(self.drop_f)
        # Input x has shape: [Batch, Dim, Frame_len, Nb_Layer]

        # Compute the key by taking a weighted sum of input across layers
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k, dim=-1)), dim=-1).transpose(1, 2)
        feature_BxTxH = self.cmp_linear_k(k) # B T H

        if self.drop_f:
            feature_BxHxT = torch.permute(feature_BxTxH, (0,2,1)) # [BxHxT]                                                              \
                                                                                                                                          
            feature_BxHxT = torch.unsqueeze(feature_BxHxT, dim=-1) # [BxHxTx1]                                                           \
                                                                                                                                          
            feature_BxHxT = self.dropout(feature_BxHxT)
            feature_BxHxT = torch.squeeze(feature_BxHxT, dim=-1) # [BxHxT]                                                                
            feature_BxTxH = torch.permute(feature_BxHxT, (0,2,1)) # [BxTxH]

        x = feature_BxTxH 

        #device = feature_BxTxH.device
        dshift = 1  # the diagonal to consider (0:includes diag, 1:from 1 over diag)                                                     \
                                                                                                                                          
        (b,n,d) = x.shape
        dcor = int(d*(d-1)/2) if dshift == 1 else int(d*(d+1)/2)
        ind = torch.triu_indices(d, d, offset=dshift).unbind()
        Ib = torch.tensor(range(b)).unsqueeze(1).repeat(1,dcor).view(-1)
        Id0 = ind[0].repeat(b)
        Id1 = ind[1].repeat(b)
        
        x = x - torch.mean(x, dim=1, keepdim=True)
        x = torch.div(x,torch.std(x, dim=1, keepdim=True)+1e-9) if dshift == 1 else x

        corr = torch.einsum('bjk,bjl->bkl', x, x/n) # (H, H)  

        corr = corr[Ib,Id0,Id1].view(b,-1)
        outs = self.pooling_fc(corr)

        return outs

class CorrelationPooling(CorrelationPoolingDrop):
    def __init__(self, head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(CorrelationPooling, self).__init__()
        p_drop=0
        self.drop_f = p_drop != 0

if __name__=='__main__':
    # back_end=MHFA(head_nb=64, inputs_dim=768, compression_dim=128, outputs_dim=256)
    back_end=CorrelationPoolingDrop(outputs_dim=256)
    x = torch.randn(5, 768, 150, 13)
    out = back_end(x)
    print(out.shape)

