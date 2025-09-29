import pandas as pd

def get_multi_label_counts(data_df, multi_label_column):
    data_df[multi_label_column] = data_df[multi_label_column].replace(999, pd.NA)
    label_counter = {}
    # iterate through the rows
    for index, row in data_df.iterrows():
        # split by comma then strip spaces
        labels = [label.strip() for label in str(row[multi_label_column]).split(',')]
        # if label not in the dict, add it
        for label in labels:
            if label == 'nan' or label == '<NA>':
                continue
            if label not in label_counter:
                label_counter[label] = 1
            else:
                label_counter[label] += 1

    # sort the dictionary by value
    sorted_label_counter = dict(sorted(label_counter.items(), key=lambda item: item[1], reverse=True))
    return sorted_label_counter