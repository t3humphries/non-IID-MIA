import numpy as np
import pandas as pd
from census_mapping import *
from config import *
import os
import pickle as pkl


def print_unique_columns_dataframe_list(dataframes):
    dataframe = pd.concat(dataframes, ignore_index=True)
    for col in dataframe:
        print(col)
        for i, df in enumerate(dataframes):
            print('{:d})     '.format(i), sorted(df[col].unique()))
        print('merge) ', sorted(dataframe[col].unique()))


def print_class_balance_per_subgroup(dataframe, subgroups, class_name='class'):
    unique_classes = np.unique(dataframe[class_name])
    unique_subgroups = np.unique(subgroups)
    subgroups = np.array(subgroups)
    for group in unique_subgroups:
        print("Group {:d} |".format(group), end=" ")
        for label in unique_classes:
            print("{:d} label {:s} |".format(np.sum(dataframe[class_name][subgroups == group] == label), str(label)), end=" ")
        print("")


def balance_classes_in_dataframe_list(dataframes, classes, nsamples=-1, verbose=False):
    """
    classes appear in order of priority: we aim for balance or more samples of classes[0]
    If nsamples=-1, we aim for the same class ratio, if nsamples is a number, every dataframe will have that number of samples"""
    v_print = print if verbose else lambda *a, **k: None
    if len(classes) != 2:
        raise ValueError("We can only balance for two classes with this function!")

    v_print("Balancing the classes")

    if nsamples > 0:
        # Find the group with the smallest number of samples from the priority class
        n0 = min([np.sum(df['class'] == classes[0]) for df in dataframes])
        n0 = min(n0, nsamples // 2)
        n1 = nsamples - n0
        balanced_dataframes = []
        for df in dataframes:
            class_0 = df[df['class'] == classes[0]].sample(n0)
            class_1 = df[df['class'] == classes[1]].sample(n1)
            df = pd.concat([class_0, class_1], ignore_index=True)
            balanced_dataframes.append(df)
    else:
        # Find the balance with percentages, not with raw samples
        # PROBLEM: we want to keep a minimum dataset size! So we are hard-coding the p0_target for now...

        # p0_target = min(max([np.sum(df['class'] == classes[0])/len(df) for df in dataframes]), 0.5)
        p0_target = 0.25
        v_print("Target percentage of class 0 samples: {:.3f}".format(p0_target))
        balanced_dataframes = []
        for i, df in enumerate(dataframes):
            s0 = np.sum(df['class'] == classes[0])
            s1 = np.sum(df['class'] == classes[1])
            if s0/s1 < p0_target:
                # We take all class 0 samples, and subsample class 1
                n0 = s0
                n1 = int(s0 * (1 - p0_target) / p0_target)
                class_0 = df[df['class'] == classes[0]].sample(n0)
                class_1 = df[df['class'] == classes[1]].sample(n1)
            else:
                n0 = int(s1 * p0_target / (1 - p0_target))
                n1 = s1
                class_0 = df[df['class'] == classes[0]].sample(n0)
                class_1 = df[df['class'] == classes[1]].sample(n1)
            v_print("For dataframe {:d}, we achieved a class balance of {:.3f}".format(i, len(class_0) / (len(class_1) + len(class_0))))
            df = pd.concat([class_0, class_1], ignore_index=True)
            balanced_dataframes.append(df)
    return balanced_dataframes


def remove_duplicates_from_dataframe_list(dataframes, verbose, seed=3):
    """Removes duplicates at random from list of dataframes, and returns the list back"""
    v_print = print if verbose else lambda *a, **k: None

    v_print("Removing random duplicates")

    # Join dataframe and assign groups
    dataframe = pd.concat(dataframes, ignore_index=True)
    subgroups = []
    for i, df in enumerate(dataframes):
        subgroups += [i] * len(df)
    subgroups = np.array(subgroups)
    np.random.seed(seed)  # This also affects Pandas
    idx = np.random.permutation(len(dataframe))

    # permute
    subgroups = subgroups[idx]
    dataframe = dataframe.iloc[idx].reset_index(drop=True)

    # remove duplicates
    dataframe_no_duplicates = dataframe.drop_duplicates(subset=None, keep='last', inplace=False, ignore_index=False)
    idx_to_keep = np.array(dataframe_no_duplicates.index)
    dataframe_no_duplicates.reset_index(drop=True, inplace=True)
    subgroups_nodup = subgroups[idx_to_keep]

    # recover each dataframe
    dataframes_nodups = []
    for group in range(len(dataframes)):
        dataframes_nodups.append(dataframe_no_duplicates[subgroups_nodup == group])

    for i in range(len(dataframes)):
        v_print('group {:d}: before {:d}, after {:d} | '.format(i, len(dataframes[i]), len(dataframes_nodups[i])), end=" ")
        v_print("class distribution: {:d} <=50K, {:d} >50K".format(sum(dataframes_nodups[i]['class'] == '<=50K'), sum(dataframes_nodups[i]['class'] == '>50K')))

    return dataframes_nodups


def preprocess_heart_dataframe(d):
    """Receives a dataframe loaded from a heart disease dataset, removes question marks and creates categorical variables
    The columns are: age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,num"""

    d.replace({'?': np.nan}, inplace=True)
    d = d.astype(float)
    d['sex'].replace({1: 'male', 0: 'female'}, inplace=True)
    d['cp'].replace({1: 'typical', 2: 'atypical', 3: 'non-anginal', 4: 'asymptomatic'}, inplace=True)
    d['fbs'].replace({0: 'no', 1: 'yes'}, inplace=True)
    d['restecg'].replace({0: 'normal', 1: 'abnormal', 2: 'hypertrophy'}, inplace=True)
    d['exang'].replace({0: 'no', 1: 'yes'}, inplace=True)
    d['slope'].replace({1: 'upsloping', 2: 'flat', 3: 'downsloping'}, inplace=True)
    d['thal'].replace({3: 'normal', 6: 'fixed', 7: 'reversible'}, inplace=True)
    d['num'].replace({0: 0, 1: 1, 2: 1, 3: 1, 4: 1}, inplace=True)

    cols_for_mean = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    for (col, vals) in d.iteritems():
        if col in cols_for_mean:
            mean_value = np.round(d[col].mean(), 1)
            d[col].fillna(value=mean_value, inplace=True)
        else:
            mode = d[col].mode()[0]
            d[col].fillna(value=mode, inplace=True)
    return d


def preprocess_students_portuguese(d):

    return d

def convert_mapping(map):
    to_return = {}
    for key in map:
        for element in map[key]:
            to_return[element] = key
    return to_return

def preprocess_kdd_census(d, year=94, all=False, balance=False):
    d = d[d.iloc[:, 40] == year]
    if all:
        attributes_mapping = {'age': 0, 'workclass': 1, 'education': 4, 'marital-status': 7, 'occupation': 9, 'race': 10, 'sex': 12, 'native-country': 34, 'class': 41}
    else:
        attributes_mapping = {'age': 0, 'workclass': 1, 'education': 4, 'marital-status': 7, 'occupation': 9, 'race': 10, 'sex': 12, 'capital-gain': 16, 'capital-loss': 17, 'native-country': 34, 'class': 41}

    # Get correct columns
    raw_dataframe = d.iloc[:, list(attributes_mapping.values())].copy()
    raw_dataframe.columns = list(attributes_mapping.keys())
    filtered_dataframe = raw_dataframe

    # Create classification
    def class_map(x):
        return '<=50K' if x == ' - 50000.' else '>50K'

    filtered_dataframe['class'] = filtered_dataframe['class'].apply(class_map)

    if balance:
        # Subsample the bigger class to get ~23% class imbalance like adult
        number_minority = sum(filtered_dataframe['class'] == '>50K')
        print("Number of samples with >50K: ", number_minority)
        sample_size = int(number_minority * (100 / 23.93) * 0.7607)
        print("We will sample: ", sample_size)
        class_0 = filtered_dataframe[filtered_dataframe['class'] == '<=50K'].sample(sample_size)
        class_1 = filtered_dataframe[filtered_dataframe['class'] == '>50K']
        df = pd.concat([class_0, class_1], ignore_index=True)
        df = df.sample(frac=1).reset_index(drop=True)
        print("After sampling, the percentage of samples <=50K is: ", sum(df['class'] == '<=50K') / len(df))
    else:
        print("No sampling to add class balance")
        df = filtered_dataframe
        print(len(df))

    # Map to new domain
    for attribute in attributes_mapping:
        if attribute not in ['age', 'capital-gain', 'capital-loss', 'class']:
            df[attribute].replace(convert_mapping(kdd_census_map[attribute]), inplace=True)
    return df

def preprocess_adult_to_match_kdd(d):
    chosen_attributes = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'capital-gain', 'capital-loss', 'native-country', 'class']
    # Remove extra columns
    raw_dataframe = d[chosen_attributes].copy()
    return raw_dataframe

def preprocess_students(df1, df2):
    """Receives two dataframes with the same columns (student-mat.csv and student-por.csv).
    Merges them deleting repeated entries."""
    pd.concat([df1, df2]).drop_duplicates().reset_index(drop=True)


def print_census_processed_statistics():
    with open('datasets/census-mix.p', 'rb') as f:
        dataset_dict = pkl.load(f)
    labels = dataset_dict['labels']
    subgroups = np.array(dataset_dict['subgroups'])

    for group in np.unique(subgroups):
        print("Group {:d} has {:d} samples".format(group, np.sum([g == group for g in subgroups])))
        labels_this_group = labels[subgroups == group]
        for label in np.unique(labels_this_group):
            print(" label {:d} in {:.1f}% of this group's samples".format(label, np.sum(labels_this_group == label) / len(labels_this_group) * 100))


if __name__ == '__main__':
    os.system('mesg n')
    from preprocess_data import import_dataset

    d = import_dataset('datasets_original/kdd_census.csv')
    print(preprocess_kdd_census(d))
    # print_census_processed_statistics()
