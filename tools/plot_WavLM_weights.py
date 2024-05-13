import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import argparse

def load_model_weights(path='../examples/voxlingua/v2/exp/final_model.pt'):
    print("Loading model...")
    model = torch.load(path, map_location=torch.device('cpu'))
    weights_k = model['speaker_extractor.back_end.weights_k']
    weights_v = model.get('speaker_extractor.back_end.weights_v', None)
    return weights_k, weights_v

def plot_weights(weights_k, weights_v=None, img_path='../examples/voxlingua/v2/exp/'):
    print("Plotting...")
    x = torch.arange(1, 14)

    mpl.rcParams.update({
        'font.size': 20,
        # 'font.family':  'serif', 
        # 'font.serif':   'CMU Serif',
        'axes.spines.top':    False, 
        'axes.spines.right':  False,
        # 'axes.prop_cycle': cycler('color', ['e41a1c', '377eb8', '4daf4a', '984ea3', 'ff7f00', 'ffff33', 'a65628', 'f781bf', '999999'])
    })
    mpl.rcParams['font.serif']=['cm']
    plt.figure(figsize=(10, 4.65))
    keys = torch.softmax(weights_k, dim=-1)
    values = torch.tensor([])
    plt.plot(x, keys, label='Keys')
    if weights_v != None:
        values = torch.softmax(weights_v, dim=-1)
        plt.plot(x, values, label='Values')
    
    print(torch.min(torch.cat((keys, values), dim=0)).item(), torch.max(torch.cat((keys, values), dim=0)).item())
    min_y, max_y = round(0, 2), round(torch.max(torch.cat((keys, values), dim=0)).item(), 2)
    tick_density = 5  # Desired number of ticks
    plt.xlabel('WawLM Layers')
    plt.ylabel('Assigned Weights')
    # plt.title('MHFA Backend')

    plt.xticks(x)
    # plt.yticks(np.linspace(min_y, max_y, tick_density))
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_path, format='pdf')

    print("Done")


if __name__ == "__main__":
    base_path = '../examples/voxlingua/v2/exp/'
    LWAP_path = base_path + 'model_5.pt'
    MHFA_path = base_path + 'final_model.pt'
    img_path = base_path + 'weights.pdf'

    # Set default values
    default_model_path = MHFA_path
    default_image_path = img_path

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", help = "Model path") # , required=True
    parser.add_argument("-out", "--imgOut", help = "Image output path") #, required=True

    args = parser.parse_args()

    model_path = args.model if args.model else default_model_path
    img_path = args.imgOut if args.imgOut else default_image_path
    
    plot_weights(*load_model_weights(model_path), img_path)
