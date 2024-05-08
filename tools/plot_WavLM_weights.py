import torch
import matplotlib.pyplot as plt
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

    plt.figure(figsize=(10, 6))
    plt.plot(x, torch.softmax(weights_k, dim=-1), label='weights_k')
    if weights_v != None:
        plt.plot(x, torch.softmax(weights_v, dim=-1), label='weights_v')

    plt.xlabel('Layers')
    plt.ylabel('Weights')
    plt.title('WawLM')

    plt.xticks(x)
    plt.legend()
    plt.savefig(img_path)

    print("Done")


if __name__ == "__main__":
    base_path = '../examples/voxlingua/v2/exp/'
    LWAP_path = base_path + 'model_5.pt'
    MHFA_path = base_path + 'final_model.pt'
    img_path = base_path + 'weights.png'

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
