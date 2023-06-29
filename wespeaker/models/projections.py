# Copyright (c) 2021 Shuai Wang (wsstriving@gmail.com)
#               2021 Zhengyang Chen (chenzhengyang117@gmail.com)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#               2023 Bing Han (hanbing97@sjtu.edu.cn)
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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_projection(conf):
    if conf['project_type'] == 'add_margin':
        projection = AddMarginProduct(conf['embed_dim'],
                                      conf['num_class'],
                                      scale=conf['scale'],
                                      margin=0.0)
    elif conf['project_type'] == 'arc_margin':
        projection = ArcMarginProduct(conf['embed_dim'],
                                      conf['num_class'],
                                      scale=conf['scale'],
                                      margin=0.0,
                                      easy_margin=conf['easy_margin'])
    elif conf['project_type'] == 'arc_margin_intertopk_subcenter':
        projection = ArcMarginProduct_intertopk_subcenter(
            conf['embed_dim'],
            conf['num_class'],
            scale=conf['scale'],
            margin=0.0,
            easy_margin=conf['easy_margin'],
            K=conf.get('K', 3),
            mp=conf.get('mp', 0.06),
            k_top=conf.get('k_top', 5),
            do_lm=conf.get('do_lm', False))
    elif conf['project_type'] == 'sphere':
        projection = SphereProduct(conf['embed_dim'],
                                   conf['num_class'],
                                   margin=4)
    elif conf['project_type'] == 'sphereface2':
        projection = SphereFace2(conf['embed_dim'],
                                 conf['num_class'],
                                 scale=conf['scale'],
                                 margin=0.0,
                                 t=conf.get('t', 3),
                                 lanbuda=conf.get('lanbuda', 0.7),
                                 margin_type=conf.get('margin_type', 'C'))
    else:
        projection = Linear(conf['embed_dim'], conf['num_class'])

    return projection


class SphereFace2(nn.Module):
    r"""Implement of sphereface2 for speaker verification:
        Reference:
            [1] Exploring Binary Classification Loss for Speaker Verification
            https://ieeexplore.ieee.org/abstract/document/10094954
            [2] Sphereface2: Binary classification is all you need
            for deep face recognition
            https://arxiv.org/pdf/2108.01513
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            lanbuda: weight of positive and negative pairs
            t: parameter for adjust score distribution
            margin_type: A:cos(theta+margin) or C:cos(theta)-margin
        Recommend margin:
            training: 0.2 for C and 0.15 for A
            LMF: 0.3 for C and 0.25 for A
        """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=32.0,
                 margin=0.2,
                 lanbuda=0.7,
                 t=3,
                 margin_type='C'):
        super(SphereFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type

        ########
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        ########

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, input, label):
        # compute similarity
        cos = F.linear(F.normalize(input), F.normalize(self.weight))

        if self.margin_type == 'A':  # arcface type
            sin = torch.sqrt(1.0 - torch.pow(cos, 2))
            cos_m_theta_p = self.scale * self.fun_g(
                torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m,
                            cos - self.mmm), self.t) + self.bias[0][0]
            cos_m_theta_n = self.scale * self.fun_g(
                cos * self.cos_m + sin * self.sin_m, self.t) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(
                1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (
                1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:  # cosface type
            cos_m_theta_p = self.scale * (self.fun_g(cos, self.t) -
                                          self.margin) + self.bias[0][0]
            cos_m_theta_n = self.scale * (self.fun_g(cos, self.t) +
                                          self.margin) + self.bias[0][0]
            cos_p_theta = self.lanbuda * torch.log(
                1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (
                1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        target_mask = input.new_zeros(cos.size())
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        cos1 = (cos - self.margin) * target_mask + cos * nontarget_mask
        output = self.scale * cos1  # for computing the accuracy
        loss = (target_mask * cos_p_theta +
                nontarget_mask * cos_n_theta).sum(1).mean()
        return output, loss

    def extra_repr(self):
        return '''in_features={}, out_features={}, scale={}, lanbuda={},
                  margin={}, t={}, margin_type={}'''.format(
            self.in_features, self.out_features, self.scale, self.lanbuda,
            self.margin, self.t, self.margin_type)


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
        """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin)  # this can make the output more continuous
        ########
        self.m = self.margin
        ########

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)
        # self.weight = self.weight
        # self.scale = self.scale

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output

    def extra_repr(self):
        return '''in_features={}, out_features={}, scale={},
                  margin={}, easy_margin={}'''.format(self.in_features,
                                                      self.out_features,
                                                      self.scale, self.margin,
                                                      self.easy_margin)


class ArcMarginProduct_intertopk_subcenter(nn.Module):
    r"""Implement of large margin arc distance with intertopk and subcenter:
        Reference:
            MULTI-QUERY MULTI-HEAD ATTENTION POOLING AND INTER-TOPK PENALTY
            FOR SPEAKER VERIFICATION.
            https://arxiv.org/pdf/2110.05042.pdf
            Sub-center ArcFace: Boosting Face Recognition by
            Large-Scale Noisy Web Faces.
            https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            scale: norm of input feature
            margin: margin
            cos(theta + margin)
            K: number of sub-centers
            k_top: number of hard samples
            mp: margin penalty of hard samples
            do_lm: whether do large margin finetune
        """

    def __init__(self,
                 in_features,
                 out_features,
                 scale=32.0,
                 margin=0.2,
                 easy_margin=False,
                 K=3,
                 mp=0.06,
                 k_top=5,
                 do_lm=False):
        super(ArcMarginProduct_intertopk_subcenter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.do_lm = do_lm

        # intertopk + subcenter
        self.K = K
        if do_lm:  # if do LMF, remove hard sample penalty
            self.mp = 0.0
            self.k_top = 0
        else:
            self.mp = mp
            self.k_top = k_top

        # initial classifier
        self.weight = nn.Parameter(
            torch.FloatTensor(self.K * out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.mmm = 1.0 + math.cos(
            math.pi - margin)  # this can make the output more continuous
        ########
        self.m = self.margin
        ########
        self.cos_mp = math.cos(0.0)
        self.sin_mp = math.sin(0.0)

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin
        self.m = self.margin
        self.mmm = 1.0 + math.cos(math.pi - margin)

        # hard sample margin is increasing as margin
        if margin > 0.001:
            mp = self.mp * (margin / 0.2)
        else:
            mp = 0.0
        self.cos_mp = math.cos(mp)
        self.sin_mp = math.sin(mp)

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input),
                          F.normalize(self.weight))  # (batch, out_dim * k)
        cosine = torch.reshape(
            cosine, (-1, self.out_features, self.K))  # (batch, out_dim, k)
        cosine, _ = torch.max(cosine, 2)  # (batch, out_dim)

        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            ########
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
            ########

        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.k_top > 0:
            # topk (j != y_i)
            _, top_k_index = torch.topk(cosine - 2 * one_hot,
                                        self.k_top)  # exclude j = y_i
            top_k_one_hot = input.new_zeros(cosine.size()).scatter_(
                1, top_k_index, 1)

            # sum
            output = (one_hot * phi) + (top_k_one_hot * phi_mp) + (
                (1.0 - one_hot - top_k_one_hot) * cosine)
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

    def extra_repr(self):
        return 'in_features={}, out_features={}, scale={}, margin={}, easy_margin={},' \
            'K={}, mp={}, k_top={}, do_lm={}'.format(
                self.in_features, self.out_features, self.scale, self.margin,
                self.easy_margin, self.K, self.mp, self.k_top, self.do_lm)


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        cos(theta) - margin
    """

    def __init__(self, in_features, out_features, scale=32.0, margin=0.20):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform_(self.weight)

    def update(self, margin):
        self.margin = margin

    def forward(self, input, label):
        # ---------------- cos(theta) & phi(theta) ---------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.margin
        # ---------------- convert label to one-hot ---------------
        one_hot = input.new_zeros(cosine.size())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', scale=' + str(self.scale) \
            + ', margin=' + str(self.margin) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        margin: margin
        cos(margin * theta)
    """

    def __init__(self, in_features, out_features, margin=2):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.margin = margin
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = nn.Parameter(torch.FloatTensor(out_features,
                                                     in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x**0, lambda x: x**1, lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x, lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x
        ]
        assert self.margin < 6

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(
            self.LambdaMin,
            self.base * (1 + self.gamma * self.iter)**(-1 * self.power))

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.margin](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.margin * theta / 3.14159265).floor()
        phi_theta = ((-1.0)**k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)
        one_hot = input.new_zeros(cos_theta.size())
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * (phi_theta - cos_theta) /
                  (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', margin=' + str(self.margin) + ')'


class Linear(nn.Module):
    """
    The linear transform for simple softmax loss
    """

    def __init__(self, emb_dim=512, class_num=1000):
        super(Linear, self).__init__()

        self.trans = nn.Sequential(nn.BatchNorm1d(emb_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(emb_dim, class_num))

    def forward(self, input, label):
        out = self.trans(input)
        return out


if __name__ == '__main__':
    #    projection = ArcMarginProduct(100,
    #                                  200,
    #                                  scale=32.0,
    #                                  margin=0.2,
    #                                  easy_margin=False)
    #
    #    print(hasattr(projection, 'update_mar'))
    projection = ArcMarginProduct_intertopk_subcenter(100,
                                                      200,
                                                      scale=32.0,
                                                      margin=0.0,
                                                      easy_margin=False,
                                                      K=3,
                                                      mp=0.06,
                                                      k_top=5)
    print(hasattr(projection, 'update'))
    projection.update(0.2)
    print(projection)
    embed = torch.randn(16, 100)
    label = torch.randint(200, (16, ))
    out = projection(embed, label)
    print(out.size())

    # for name, param in projection.named_parameters():
    #     print(name)
    #     print(param.shape)
