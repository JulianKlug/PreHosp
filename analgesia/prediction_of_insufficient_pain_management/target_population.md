# Target population

Target population: primary missions of adult trauma patients that start with a VAS > 3
data_df = data_df[data_df["VAS_on_scene"] > 3]
data_df = data_df.drop_duplicates(subset=["SNZ Ereignis Nr. "])

if restrict_to_trauma:
    n_non_trauma = data_df[data_df['Einteilung (reduziert)'] != 'Unfall'].shape[0]
    print(f'Excluded {n_non_trauma} non-trauma patients')

    # adult non-trauma patients
    n_adult_non_trauma = data_df[(data_df['Einteilung (reduziert)'] != 'Unfall') & (data_df["Alter "] >= 16)].shape[0]
    print(f'Excluded {n_adult_non_trauma} adult non-trauma patients')
    # pediatric non-trauma patients
    n_pediatric_non_trauma = data_df[(data_df['Einteilung (reduziert)'] != 'Unfall') & (data_df["Alter "] < 16)].shape[0]
    print(f'Excluded {n_pediatric_non_trauma} pediatric non-trauma patients')

    data_df = data_df[data_df['Einteilung (reduziert)'] == 'Unfall']

if restrict_to_primary:
    n_secondary = data_df[data_df['Einsatzart'] != 'Primär'].shape[0]
    print(f'Excluded {n_secondary} secondary transport patients')

    # adult secondary transport patients
    n_adult_secondary = data_df[(data_df['Einsatzart'] != 'Primär') & (data_df["Alter "] >= 16)].shape[0]
    print(f'Excluded {n_adult_secondary} adult secondary transport patients')
    # pediatric secondary transport patients
    n_pediatric_secondary = data_df[(data_df['Einsatzart'] != 'Primär') & (data_df["Alter "] < 16)].shape[0]
    print(f'Excluded {n_pediatric_secondary} pediatric secondary transport patients')
    data_df = data_df[data_df['Einsatzart'] == 'Primär']

adult_df = data_df[data_df["Alter "] >= 16]
