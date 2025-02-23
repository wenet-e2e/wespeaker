# Copyright (c) 2025 Shuai Wang (wsstriving@gmail.com)
#               2025 Junjie LI (junjie98.li@connect.polyu.hk)
#               2025 Tianchi Liu (tianchi_liu@u.nus.edu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''The implementation of Xi_vector.

Reference:
[1] Lee, K. A., Wang, Q., & Koshinaka, T. (2021). Xi-vector embedding
for speaker recognition. IEEE Signal Processing Letters, 28, 1385-1389.
'''


import torch
import wespeaker.models.ecapa_tdnn as ecapa_tdnn
import wespeaker.models.tdnn as tdnn




def XI_VEC_ECAPA_TDNN_c1024(feat_dim, embed_dim, pooling_func='XI', emb_bn=False):
    return ecapa_tdnn.ECAPA_TDNN(channels=1024,
                                 feat_dim=feat_dim,
                                 embed_dim=embed_dim,
                                 pooling_func=pooling_func,
                                 emb_bn=emb_bn)


def XI_VEC_ECAPA_TDNN_c512(feat_dim, embed_dim, pooling_func='XI', emb_bn=False):
    return ecapa_tdnn.ECAPA_TDNN(channels=512,
                                 feat_dim=feat_dim,
                                 embed_dim=embed_dim,
                                 pooling_func=pooling_func,
                                 emb_bn=emb_bn)



def XI_VEC_XVEC(feat_dim, embed_dim, pooling_func='XI'):
    return tdnn.XVEC(feat_dim=feat_dim, embed_dim=embed_dim, pooling_func=pooling_func)


if __name__ == '__main__':
    x = torch.rand(1, 200, 80)
    model = XI_VEC_XVEC(feat_dim=80, embed_dim=512, pooling_func='XI')
    model.eval()
    y = model(x)
    print(y[-1].size())

    num_params = sum(p.numel() for p in model.parameters())
    print("{} M".format(num_params / 1e6))

    from thop import profile
    x_np = torch.randn(1, 200, 80)
    flops, params = profile(model, inputs=(x_np, ))
    print("FLOPs: {} G, Params: {} M".format(flops / 1e9, params / 1e6))
