import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
# from sklearn.model_selection import train_test_split

def add_length_cat(df, categories=7):
    categories = min(int(df.size*0.2/(categories-2) + 1), categories)
    # print(df['subdir'].iloc[0],"split_size:", df.size, "categories:", categories)
    bin_edges = [df['length'].quantile(i) for i in np.linspace(0, 1.0, categories)]
    bin_edges[0] -= 1; bin_edges[-1] += 1

    df['length_cat'] = pd.cut(df['length'], bins=bin_edges, labels=range(1, len(bin_edges)))
    return df

# def split_data_frame(df):
#     train, test = train_test_split(df, test_size=0.2, stratify=df['length_cat'])
#     return train, test

# def split_grouped_dataset(grouped):
#     train_set = []; test_set = []
#     for _, group_data in grouped:
#         group_data = add_length_cat(group_data)
#         train, test = split_data_frame(group_data)
#         train_set.append(train)
#         test_set.append(test)
#     train_out = pd.concat(train_set)
#     test_out = pd.concat(test_set)
    
#     return train_out.sample(frac=1), test_out.sample(frac=1)

# def split_dataset(in_path, out_train_path, out_test_path):
#     data = pd.read_csv(in_path)
#     filtered = data[~data['subdir'].isin(['5', '7']) ]
#     grouped = filtered.groupby('subdir')

#     train, test = split_grouped_dataset(grouped)
#     train.to_csv(out_train_path, index=False)
#     test.to_csv(out_test_path, index=False)

def print_stats(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train_mean = train['length'].mean()
    test_mean = test['length'].mean()

    train_group_means = train.groupby('subdir')['length'].mean()
    test_group_means = test.groupby('subdir')['length'].mean()

    group_means = pd.merge(train_group_means, test_group_means, on='subdir')
    group_means.columns = ['train_mean_length', 'test_mean_length']

    print(group_means, "\n")
    print("Global mean length:")
    print("train:", train_mean, "test:", test_mean)

def plot_histogram(train_counts, test_counts, img_path, img_name, x_label):
    mpl.rcParams.update({
        # 'font.size': 16,
        # 'font.family':  'serif', 
        # 'font.serif':   'CMU Serif',
        'axes.spines.top':    False, 
        'axes.spines.right':  False,
        # 'axes.prop_cycle': cycler('color', ['e41a1c', '377eb8', '4daf4a', '984ea3', 'ff7f00', 'ffff33', 'a65628', 'f781bf', '999999'])
    })
    plt.figure(figsize=(10, 4))
    plt.bar(train_counts.index, train_counts.values, label='Train')
    plt.bar(test_counts.index, test_counts.values, label='Test')
    plt.xlabel(x_label)
    plt.ylabel('Records')
    # plt.title('Histogram of Records by Dialect')
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_path + img_name + '.pdf', format='pdf')
    # plt.show()

def plot_stats(train_path, test_path, img_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    train_count = train['subdir'].value_counts()
    test_count = test['subdir'].value_counts()
    category_sizes = pd.DataFrame({'train_category_size': train_count, 'test_category_size': test_count})
    category_sizes['percentage'] = category_sizes['test_category_size'] / (category_sizes['test_category_size'] + category_sizes['train_category_size'])
    category_sizes = category_sizes.nsmallest(10, 'test_category_size')
    print(category_sizes)
    # plot_histogram(train_count, test_count, img_path, 'category_sizes', 'Dialect')


    train = add_length_cat(train, 7)
    train = train[train['subdir'] == '1-3']
    test = add_length_cat(test, 7)
    test = test[test['subdir'] == '1-3']

    train_count = train['length_cat'].value_counts()
    test_count = test['length_cat'].value_counts()
    num_of_length_cat = pd.DataFrame({'train_lenght_cat': train_count, 'test_lenght_cat': test_count})
    num_of_length_cat['percentage'] = num_of_length_cat['test_lenght_cat'] / (num_of_length_cat['test_lenght_cat'] + num_of_length_cat['train_lenght_cat'])
    plot_histogram(train_count, test_count, img_path, 'num_of_length_cat', 'Length Category')

    print(num_of_length_cat)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("-d", "--dataset", help = "Dataset path") # , required=True
    parser.add_argument("-train", "--train", help = "Train output path") #, required=True
    parser.add_argument("-test", "--test", help = "Test output path") #, required=True
    parser.add_argument("-vl", "--voxlingua", action="store_true", help = "Runs voxlingua") #, required=True

    args = parser.parse_args()

    # # Set constants
    BASE_PATH = '../examples/voxlingua/v2/exp/'
    NAKI = False if args.voxlingua else True
    # DATASET_FILE = 'naki_recordings.csv' if NAKI else 'recording_lengths.csv'
    DATASET_NAME = 'naki' if NAKI else 'voxlingua'

    # default_dataset_path = BASE_PATH + DATASET_FILE
    default_train_path = BASE_PATH + DATASET_NAME + '_train.csv'
    default_test_path = BASE_PATH + DATASET_NAME + '_test.csv'
    default_img_path = BASE_PATH

    # dataset_path = args.dataset if args.dataset else default_dataset_path
    train_path = args.train if args.train else default_train_path
    test_path = args.test if args.test else default_test_path

    # split_dataset(dataset_path, train_path, test_path)
    # print_stats(train_path, test_path)
    plot_stats(train_path, test_path, default_img_path)
