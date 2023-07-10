import os
import torch.distributed as dist


def getoneNode():
    # get one node from slurm cluster as the host address
    nodelist = os.environ['SLURM_JOB_NODELIST']
    nodelist = nodelist.strip().split(',')[0]
    import re
    text = re.split('[-\\[\\]]', nodelist)
    if ('' in text):
        text.remove('')
    return text[0] + '-' + text[1] + '-' + text[2]


def init_ddp(gpu_list='[0]', port=23456):
    if isinstance(gpu_list, int):
        gpu_list = [gpu_list]

    if 'WORLD_SIZE' in os.environ:
        # using torchrun to start:
        # egs: torchrun --standalone --rdzv_endpoint=localhost:$PORT_k \
        #                            --nnodes=1 \
        #                            --nproc_per_node=2 \
        #                            train.py \
        #                            --config conf/config.yaml 
        #                            --gpu_list '[0,1]'
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu_id = int(gpu_list[rank])
    elif 'SLURM_LOCALID' in os.environ:
        # start process using slurm
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        gpu_id = local_rank

        os.environ['MASTER_ADDR'] = getoneNode()
        os.environ['MASTER_PORT'] = str(port)
    else:
        # run locally with only one process
        rank = 0
        local_rank = 0
        world_size = 1
        gpu_id = int(gpu_list[rank])

        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(port)

    # env:// means we use the environment variables to
    # set the host_addr and port
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)

    return gpu_id
