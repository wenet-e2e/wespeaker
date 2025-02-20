
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
