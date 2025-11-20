import pandas as pd
import numpy as np


def verify_if_diagnosis_correct(pre_hospital_diagnosis, in_hospital_diagnosis):
    # diagnosis is either a number or a string with a list of numbers separated by commas
    # replace NaN with 0 for both diagnoses
    if pd.isna(pre_hospital_diagnosis):
        pre_hospital_diagnosis = 0
    if pd.isna(in_hospital_diagnosis):
        in_hospital_diagnosis = 0

    if pre_hospital_diagnosis == in_hospital_diagnosis:
        return True
    
    # convert in_hospital_diagnosis to a list if it's a string
    if isinstance(in_hospital_diagnosis, str):
        in_hospital_diagnosis = [int(x.strip()) for x in in_hospital_diagnosis.split(',')]
    else:
        in_hospital_diagnosis = [in_hospital_diagnosis]
    # convert pre_hospital_diagnosis to a list if it's a string
    if isinstance(pre_hospital_diagnosis, str):
        pre_hospital_diagnosis = [int(x.strip()) for x in pre_hospital_diagnosis.split(',')]
    else:
        pre_hospital_diagnosis = [pre_hospital_diagnosis]

    # Polytrauma	9
    # if any of the diagnoses is polytrauma, if the other contains multiple diagnoses, we consider it correct
    if 9 in pre_hospital_diagnosis and isinstance(in_hospital_diagnosis, list) and len(in_hospital_diagnosis) > 1:
        return True
    if 9 in in_hospital_diagnosis and isinstance(pre_hospital_diagnosis, list) and len(pre_hospital_diagnosis) > 1:
        return True

    # if any element of pre_hospital_diagnosis is in in_hospital_diagnosis, return True
    if isinstance(pre_hospital_diagnosis, list) and isinstance(in_hospital_diagnosis, list):
        for diag in pre_hospital_diagnosis:
            if diag in in_hospital_diagnosis:
                return True
    
    return False



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


def preprocess_diagnoses(data_df):
    # Preprocess diagnoses
    data_df['Main diagnosis pre-hospital'] = data_df['Main diagnosis pre-hospital'].replace('<NA>', np.nan)
    data_df['Main diagnosis pre-hospital'] = data_df['Main diagnosis pre-hospital'].replace('Vd. a. Asphiktische REA', 10)
    data_df['Main diagnosis pre-hospital'] = data_df['Main diagnosis pre-hospital'].replace('1. CO Intoxikation durch Rauchgasvergiftung (Kachelofen)\n   - CO 20%\n   - Schwindel, Unwohlsein, fragliche krampfartigen Äquivalente', 0)
    data_df['Main diagnosis pre-hospital'] = data_df['Main diagnosis pre-hospital'].replace('1. CO INtoxikation durch Rauchgasvergiftung (Kachelofen) mit\n   - Krampfäquivalent, Schwindel, Übelkeit\n   - CO 22%', 0)

    data_df['Main diagnosis in-hospital'] = data_df['Main diagnosis in-hospital'].replace('Obstrukt.Atemversagen -REA', 10)
    data_df['Main diagnosis in-hospital'] = data_df['Main diagnosis in-hospital'].replace('C2-Intoxikation', 0)

    data_df['prehospital_diagnosis_correct'] = data_df.apply(
        lambda row: verify_if_diagnosis_correct(
            row['Main diagnosis pre-hospital'],
            row['Main diagnosis in-hospital']
        ),
        axis=1
    )

    return data_df