import pandas as pd

def convert(path, index_map):
    df = pd.read_pickle(path, compression='infer', storage_options=None)
    # print(df.index, str(df.index[1]), type(str(df.index[1])))
    # print(df.index.map(index_map).fillna(df.index))
    
    # df = df.applymap(lambda x: round((x * 100), 2))

    print(df.loc['accuracy'])
    df.loc['accuracy', ['precision', 'recall', 'support']] = [None, None, df.loc['macro avg', ['support']][0]]
    df.index = df.index.astype(str)

    columns = ['precision', 'recall', 'f1-score']
    for column in columns:
        df[column] = df[column].apply(lambda x: '{:.2f}'.format(round(x * 100, 2) if x else ''))
    df['support'] = df['support'].apply(lambda x: int(x))
    df.index = [f'{idx} - {index_map[idx]}' if idx in index_map else idx for idx in df.index]
    return df.style.to_latex()

def create_table(base_path, name, index_map):
    latex = convert(base_path + name + '.pkl', index_map)
    with open(base_path + name + "_tab.txt", "w") as text_file:
        text_file.write(latex)

## Oblasti
regions = {
    "1": "Česká",
    "2": "Středomoravská",
    "3": "Východomoravská",
    "4": "Slezkomoravská",
}

## Podoblasti
subregions = {
    "1-1": "Severovýchodočeská",
    "1-2": "Středočeská",
    "1-3": "Jihozápadočeská",
    "1-4": "Českomoravská",
    "2-1": "Jižní",
    "2-2": "Západní",
    "2-3": "Východní",
    "2-4": "Centrální",
    "3-1": "Slovácko ",
    "3-2": "Zlinsko",
    "3-3": "Valašsko",
    "4-1": "Slezskomoravská",
    "4-2": "Slezskopolská",
}

if __name__ == "__main__":
    base_path = '../examples/voxlingua/v2/exp/'
    name = 'report_subgroup'

    create_table(base_path, 'report_subgroup', subregions)
    create_table(base_path, 'report', regions)