import os
import logging
import torch
import numpy as np
import random
import yaml
from scipy import signal


def genlogger(outdir, fname):
    formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
    logging.basicConfig(
            level=logging.DEBUG,
            format="[ %(levelname)s : %(asctime)s ] - %(message)s")
    logger = logging.getLogger("Pyobj, f")
    # Dump log to file
    fh = logging.FileHandler(os.path.join(outdir, fname))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_config_or_kwargs(config_file, **kwargs): 
    """parse_config_or_kwargs

    :param config_file: Config file that has parameters, yaml format
    :param **kwargs: Other alternative parameters or overwrites for config
    """
    with open(config_file) as con_read:
        yaml_config = yaml.load(con_read, Loader=yaml.FullLoader)
    # values from config file are all possible params
    help_str = "Valid Parameters are:\n"
    help_str += "\n".join(list(yaml_config.keys()))
    # passed kwargs will override yaml config
    #for key in kwargs.keys():
        #assert key in yaml_config, "Parameter {} invalid!\n".format(key) + help_str
    return dict(yaml_config, **kwargs)


def validate_path(dir_name):
    """ Create the directory if it doesn't exist
    :param dir_name
    :return: None
    """
    dir_name = os.path.dirname(dir_name)  # get the path
    if not os.path.exists(dir_name) and (dir_name != ''):
        os.makedirs(dir_name)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_scp(scp_file):
    key_value_list = []
    with open(scp_file,"r") as fp:
        line = fp.readline()
        while line:
            tokens = line.strip().split()
            key = tokens[0]
            value = tokens[1]
            key_value_list.append((key, value))
            line = fp.readline()
    return key_value_list


def spk2id(utt_spk_list):
    spk_set=set()
    for utt_spk in utt_spk_list:
        spk_set.add(utt_spk[1])
    spk_list=list(spk_set)

    spk2id_dict = {}
    spk_list.sort()
    for i, spk in enumerate(spk_list):
        spk2id_dict[spk] = i
    return spk2id_dict


def criterion_improver(mode):
    """Returns a function to ascertain if criterion did improve
    
    :mode: can be ether 'loss' or 'acc'
    :returns: function that can be called, function returns true if criterion improved
    """
    assert mode in ('loss', 'acc')
    best_value = np.inf if mode == 'loss' else 0

    def comparator(x, best_x):
        return x < best_x if mode == 'loss' else x > best_x

    def inner(x):
        # rebind parent scope variable
        nonlocal best_value
        if comparator(x, best_value):
            best_value = x
            return True
        return False
    return inner


def get_random_chunk(data, chunk_len):
    # chunking: randomly select a range of size min(chunk_len, len).
    data_len = len(data)
    data_shape = data.shape
    adjust_chunk_len = min(data_len, chunk_len)
    chunk_start = random.randint(0, data_len - adjust_chunk_len)

    data = data[chunk_start:chunk_start+adjust_chunk_len]
    # padding if needed
    if adjust_chunk_len < chunk_len:
        chunk_shape = chunk_len if len(data_shape)==1 else (chunk_len, data.shape[1])
        data = np.resize(data, chunk_shape) # repeating

    return data


## Adapted from wenet implementation
def spec_augmentation(x,
            warp_for_time=False,
            num_t_mask=1,
            num_f_mask=1,
            max_t=80,
            max_f=20,
            max_w=80,
            prob=0.6):
    """ do spec augmentation on x

    Args:
        x: input feature, T * F 2D
        num_t_mask: number of time mask to apply
        num_f_mask: number of freq mask to apply
        max_t: max width of time mask
        max_f: max width of freq mask
        max_w: max width of time warp

    Returns:
        augmented feature (x)
    """
    if random.random() > prob:
        return x

    y = x #np.copy(x)
    max_frames = y.shape[0]
    max_freq = y.shape[1]

    # time warp
    if warp_for_time and max_frames > max_w * 2:
        center = random.randrange(max_w, max_frames - max_w)
        warped = random.randrange(center - max_w, center + max_w) + 1

        left = Image.fromarray(x[:center]).resize((max_freq, warped), BICUBIC)
        right = Image.fromarray(x[center:]).resize(
            (max_freq, max_frames - warped), BICUBIC)
        y = np.concatenate((left, right), 0)
    # time mask
    #max_t = min(max_t, max_frames//5)
    for i in range(num_t_mask):
        """
        length = random.randint(1, max_t)
        start = random.randint(0, max_frames - length)
        end = start + length
        """
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        y[start:end, :] = 0
    # freq mask
    for i in range(num_f_mask):
        """
        length = random.randint(1, max_f)
        start = random.randint(0, max_freq - length)
        end = start + length
        """
        start = random.randint(0, max_freq - 1)
        length = random.randint(1, max_f)
        end = min(max_freq, start + length)
        y[:, start:end] = 0
    return y


def speed_perturb(x, speed_perturb_idx=0):
    speed_list = [1.0, 0.9, 1.1]
    speed = speed_list[speed_perturb_idx]

    x = x.astype(np.float32)
    y = signal.resample(x, int(len(x)/speed))

    return y
