# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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

import wespeaker.models.tdnn as tdnn
import wespeaker.models.ecapa_tdnn as ecapa_tdnn
import wespeaker.models.resnet as resnet
import wespeaker.models.repvgg as repvgg
import wespeaker.models.campplus as campplus
import wespeaker.models.eres2net as eres2net
import wespeaker.models.res2net as res2net


def get_speaker_model(model_name: str):
    if model_name.startswith("XVEC"):
        return getattr(tdnn, model_name)
    elif model_name.startswith("ECAPA_TDNN"):
        return getattr(ecapa_tdnn, model_name)
    elif model_name.startswith("ResNet"):
        return getattr(resnet, model_name)
    elif model_name.startswith("REPVGG"):
        return getattr(repvgg, model_name)
    elif model_name.startswith("CAMPPlus"):
        return getattr(campplus, model_name)
    elif model_name.startswith("ERes2Net"):
        return getattr(eres2net, model_name)
    elif model_name.startswith("Res2Net"):
        return getattr(res2net, model_name)
    else:  # model_name error !!!
        print(model_name + " not found !!!")
        exit(1)
