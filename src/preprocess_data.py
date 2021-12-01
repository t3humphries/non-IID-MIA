import csv
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
from config import *
import utils
import utils
from pre_process_texas import read_texas_data


def import_dataset(filename, sep=','):
    dataframe = pd.read_csv(filename, sep=sep)
    return dataframe

def encode_dataset(dataframe, binary_encoding=True, class_name='class'):

    def normalize_dataset(X):
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)

    # convert class to ints
    dataframe[class_name] = dataframe[class_name].astype('category')
    dataframe[class_name] = dataframe[class_name].cat.codes
    # Split off the labels
    labels = dataframe[class_name].values
    dataframe.drop(columns=class_name, inplace=True)

    # get categorical attributes
    columns = dataframe.columns
    numeric_columns = dataframe._get_numeric_data().columns
    categorical_attributes = list(set(columns) - set(numeric_columns))

    if binary_encoding:
        #print('Converting categorial to numeric using a one hot encoding')
        binary_attributes = []
        other_attributes = []
        for attribute in categorical_attributes:
            if len(dataframe[attribute].unique()) == 2:
                binary_attributes.append(attribute)
            else:
                other_attributes.append(attribute)

        numeric_dataset = pd.get_dummies(dataframe, columns=other_attributes)
        numeric_dataset[binary_attributes] = numeric_dataset[binary_attributes].astype(
            'category')
        numeric_dataset[binary_attributes] = numeric_dataset[binary_attributes].apply(
            lambda x: x.cat.codes)
    else:
        #print('Converting categorial to numeric using an integer encoding')
        numeric_dataset = dataframe.copy()
        numeric_dataset[categorical_attributes] = numeric_dataset[categorical_attributes].astype(
            'category')
        numeric_dataset[categorical_attributes] = numeric_dataset[categorical_attributes].apply(
            lambda x: x.cat.codes)
    # convert to np array
    features = numeric_dataset.values
    features = normalize_dataset(features)
    return features, labels, list(numeric_dataset.columns.values)


def cluster_split(dataset, labels, n_clusters=2, random_seed=0):
    subgroups = -np.ones(len(labels), dtype=int)
    for label in np.unique(labels):
        indices, data_this_label = zip(*[(idx, row) for idx, row in enumerate(dataset) if labels[idx] == label])
        clusters = KMeans(n_clusters=n_clusters, random_state=random_seed).fit(data_this_label)
        subgroups_this_label = clusters.labels_
        subgroups[list(indices)] = subgroups_this_label
        # print("Done cluster split for label {:d}".format(label))
    if any(subgroups < 0):
        raise ValueError("This should not happen!")
    return list(subgroups)


def generate_processed_dataset(dataset_name):
    def adult_education_num_split(dataframe):
        mask = (dataframe['education-num'] <= 10)
        dataframe.drop(columns=['education', 'education-num'], inplace=True)
        return dataframe, mask

    def adult_sex_split(dataframe):
        mask = (dataframe['sex'] == 'Male')
        dataframe.drop(columns='sex', inplace=True)
        return dataframe, mask

    def students_school_split(dataframe):
        mask = (dataframe['school'] == 'GP')
        dataframe.drop(columns='school', inplace=True)
        return dataframe, mask
    
    def texas_region_3_split(dataframe):
        mask = (dataframe['PUBLIC_HEALTH_REGION'] == '03')
        dataframe.drop(columns='PUBLIC_HEALTH_REGION', inplace=True)
        return dataframe, mask

    if dataset_name.startswith('adult'):
        dataframe = import_dataset(os.path.join(ORIGINAL_DATASET_PATH, 'dataset_adult_no_missing.csv'))
        if dataset_name.startswith('adult-clus'):
            _, aux1, aux2 = dataset_name.split('-')
            nclusters = int(aux1[4:])
            random_seed = int(aux2)
            features, labels, columns = encode_dataset(dataframe)
            subgroups = cluster_split(features, labels, n_clusters=nclusters, random_seed=random_seed)
        elif dataset_name.startswith('adult-numedu'):
            dataframe, mask = adult_education_num_split(dataframe)
            features, labels, columns = encode_dataset(dataframe)
            subgroups = [0 if val else 1 for val in mask]
        elif dataset_name.startswith('adult-sex'):
            dataframe, mask = adult_sex_split(dataframe)
            features, labels, columns = encode_dataset(dataframe)
            if dataset_name == 'adult-sex':
                subgroups = [0 if val else 1 for val in mask]
            elif dataset_name == 'adult-sex-inv':
                subgroups = [1 if val else 0 for val in mask]
            else:
                raise ValueError("Dataset: ".format(dataset_name))
        else:
            raise ValueError(dataset_name)
    elif dataset_name.startswith('compas'):
        raw_dataframe = import_dataset(os.path.join(ORIGINAL_DATASET_PATH, 'compas-scores-two-years.csv'))
        chosen_columns = ['age', 'c_charge_degree', 'race', 'age_cat', 'sex',
                          'priors_count', 'days_b_screening_arrest', 'is_recid', 'two_year_recid']
        raw_dataframe = raw_dataframe[chosen_columns]
        propublica_mask = (raw_dataframe['days_b_screening_arrest'] <= 30) & (
                raw_dataframe['days_b_screening_arrest'] >= -30) & (raw_dataframe['is_recid'] != -1) & (raw_dataframe['c_charge_degree'] != 'O')
        dataframe = raw_dataframe[propublica_mask].copy()
        if dataset_name.startswith('compas-clus'):
            _, aux1, aux2 = dataset_name.split('-')
            nclusters = int(aux1[4:])
            random_seed = int(aux2)
            features, labels, columns = encode_dataset(dataframe, class_name='two_year_recid')
            subgroups = cluster_split(features, labels, n_clusters=nclusters, random_seed=random_seed)
        else:
            raise ValueError(dataset_name)
    elif dataset_name == 'heart-mix':
        heart_datasets = ['cleveland', 'hungarian', 'switzerland', 'va']
        dataframes = []
        subgroups = []
        for i, name in enumerate(heart_datasets):
            raw_dataframe = import_dataset(os.path.join(ORIGINAL_DATASET_PATH, 'heart_{:s}.csv'.format(name)))
            d = utils.preprocess_heart_dataframe(raw_dataframe)
            subgroups += [i] * len(d)
            dataframes.append(d)
        dataframe = pd.concat(dataframes, ignore_index=True)
        features, labels, columns = encode_dataset(dataframe, class_name='num')
    elif dataset_name.startswith('students'):
        dataframe = import_dataset(os.path.join(ORIGINAL_DATASET_PATH, 'student-por.csv'), sep=';')
        if not dataset_name.endswith('full'):
            dataframe.drop(columns='G1', inplace=True)
            dataframe.drop(columns='G2', inplace=True)
        dataframe, mask = students_school_split(dataframe)
        subgroups = [0 if val else 1 for val in mask]
        if dataset_name.startswith('students2'):
            dataframe['G3'] = pd.cut(dataframe['G3'], bins=[-1, 9, 20], labels=['fail', 'pass'])
        elif dataset_name.startswith('students5'):
            dataframe['G3'] = pd.cut(dataframe['G3'], bins=[-1, 9, 11, 13, 15, 20], labels=['F', 'D', 'C', 'B', 'A'])
        else:
            raise NotImplementedError
        utils.print_class_balance_per_subgroup(dataframe, subgroups, class_name='G3')
        features, labels, columns = encode_dataset(dataframe, class_name='G3')
    elif dataset_name in ('kdd-census2', 'kdd-census2-bal'):
        census_dataset = import_dataset(os.path.join(ORIGINAL_DATASET_PATH, 'kdd_census.csv'))
        census_94 = utils.preprocess_kdd_census(census_dataset, year=94)
        adult_dataset = import_dataset(os.path.join(ORIGINAL_DATASET_PATH, 'dataset_adult_no_missing.csv'))
        adult_dataframe = utils.preprocess_adult_to_match_kdd(adult_dataset)

        dataframes = [census_94, adult_dataframe]
        utils.print_unique_columns_dataframe_list(dataframes)
        dataframes = utils.remove_duplicates_from_dataframe_list(dataframes, verbose=True, seed=3)
        if dataset_name == 'kdd-census2-bal':
            dataframes = utils.balance_classes_in_dataframe_list(dataframes, classes=['>50K', '<=50K'], verbose=True)
        subgroups = []
        for i, df in enumerate(dataframes):
            subgroups += [i] * len(df)
        dataframe = pd.concat(dataframes, ignore_index=True)
        print("Final dataframes:")
        for i, df in enumerate(dataframes):
            print('group {:d}: {:d} | '.format(i, len(df)), end=" ")
            print("Class balance: {:d} <=50K, {:d} >50K".format(sum(df['class'] == '<=50K'), sum(df['class'] == '>50K')))

        features, labels, columns = encode_dataset(dataframe, class_name='class')

    elif dataset_name.startswith('texas'):
        texas_dataframe = read_texas_data()
        if dataset_name == 'texas-region-3':
            dataframe, mask = texas_region_3_split(texas_dataframe)
            features, labels, columns = encode_dataset(dataframe)
            subgroups = [0 if val else 1 for val in mask]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    assert len(subgroups) == len(labels)
    dataset_dict = {
        'features': np.array(features).astype(np.float32),
        'labels': np.array(labels).astype(np.int32),
        'columns': columns,
        'subgroups': subgroups
    }

    return dataset_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, nargs='?', default='texas-region-group')
    args = parser.parse_args()

    if not os.path.exists(ORIGINAL_DATASET_PATH):
        raise ValueError("PREPROCESS_DATA: Path {:s} does not exist!".format(ORIGINAL_DATASET_PATH))
    if not os.path.exists(PROCESSED_DATASET_PATH):
        os.mkdir(PROCESSED_DATASET_PATH)
    assert args.dataset.startswith(('adult', 'compas', 'heart', 'kdd', 'students', 'texas'))
    if os.path.exists(os.path.join(PROCESSED_DATASET_PATH, args.dataset + '.p')):
        print("PREPROCESS_DATA: Dataset {:s} already exists, exiting...".format(args.dataset))
    else:
        dataset_dict = generate_processed_dataset(args.dataset)
        with open(os.path.join(PROCESSED_DATASET_PATH, args.dataset + '.p'), 'wb') as f:
            pickle.dump(dataset_dict, f)
        print("PREPROCESS_DATA: Dataset {:s} created!".format(args.dataset))


if __name__ == '__main__':
    os.system('mesg n')
    main()
