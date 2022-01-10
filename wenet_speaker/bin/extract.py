# coding=utf-8
#!/usr/bin/env python3
import os
import kaldiio
import fire, yaml
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from wenet_speaker.models import *
from wenet_speaker.utils.utils import *
from wenet_speaker.dataset.dataset import FeatList_LableDict_Dataset
from wenet_speaker.utils.checkpoint import load_checkpoint


def extract(config='conf/config.yaml', **kwargs):

    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)

    model_path = configs['model_path']
    data_scp = configs['data_scp']
    embed_ark = configs['embed_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 2)
    raw_wav = configs.get('raw_wav', True)
    feat_dim = configs['feature_args'].get('feat_dim', 40)

    # Since the input length is not fixed, we set the built-in cudnn auto-tuner to False
    torch.backends.cudnn.benchmark = False 

    model = eval(configs['model'])(**configs['model_args'])
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()
    
    # prepare dataset and dataloader
    utt_wav_list = read_scp(data_scp)
    dataset = FeatList_LableDict_Dataset(utt_wav_list, whole_utt=(batch_size==1), raw_wav=raw_wav, feat_dim=feat_dim)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, prefetch_factor=4)

    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"

    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:'+embed_ark+","+embed_scp) as writer:
            t_bar = tqdm(ncols=100, total=len(dataloader), desc=('extract_embed: '))
            for i, (utts, feats, _) in enumerate(dataloader): 
                t_bar.update()

                feats = feats.float().to(device) # (B,T,F)
                # Forward through model
                outputs = model(feats) # (embed_a, embed_b) in most cases
                embeds = outputs[0] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy() #(B,F)
                
                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    writer(utt, embed)

            t_bar.close()


if __name__ == '__main__':
    fire.Fire(extract)
