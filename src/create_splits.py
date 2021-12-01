import pickle
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os
from config import *

np.set_printoptions(precision=3)


def parse_split_name(name):
    """The formats are:
    n{:d}_g{:d}
    n{:d}_p{:.2f}
    nf{:d}_g{:d}"""
    if name.endswith('_rand'):
        rand_flag = True
        name = name[:-5]
    else:
        rand_flag = False
    nstr, gstr = name.split('_')
    if nstr.startswith('nf'):
        n, full_flag = int(nstr[2:]), True
    else:
        n, full_flag = int(nstr[1:]), False
    if gstr.startswith('p'):
        val = float(gstr[1:])
    else:
        val = int(gstr[1:])
    return n, val, full_flag, rand_flag


def check_sample_size_feasibility(train, test, nonmemb, total_group_sizes):

    aux = np.array(total_group_sizes) - np.array(train) - np.array(test)
    if any(aux < 0):
        raise ValueError("The dataset is not big enough for the train+test sample sizes", train, test, total_group_sizes)
    aux2 = np.array(total_group_sizes) - np.array(train) - np.array(nonmemb)
    if any(aux2 < 0):
        raise ValueError("The dataset is not big enough for the memb+nonmemb sample sizes", train, nonmemb, total_group_sizes)
    return 1


def create_split_dict(run, ntrain, ntest, nnonmemb, idx_per_group, nsamples, min_shadow_data=3000, verbose=True):

    group_sizes = [len(val) for val in idx_per_group]
    check_sample_size_feasibility(ntrain, ntest, nnonmemb, group_sizes)
    rng = np.random.default_rng(seed=run)

    train_idx = []
    test_idx = []
    non_member_idx = []
    shadow_data_idx = []
    for group, (ntrain, ntest, nnmemb) in enumerate(zip(ntrain, ntest, nnonmemb)):
        idx_perm_this_group = list(rng.permutation(idx_per_group[group]))

        train_idx += idx_perm_this_group[:ntrain]
        test_idx += idx_perm_this_group[ntrain: ntrain + ntest]
        non_member_idx += idx_perm_this_group[ntrain: ntrain + nnmemb]  # Overlaps with test_idx, doesn't matter, we maximize shadow data this way
        shadow_data_idx += idx_perm_this_group[ntrain + nnmemb:][:nsamples]  # This ensures we do not give more than nsamples per group as shadow data

    # Sanity check
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(train_idx) & set(non_member_idx)) == 0
    assert len(set(shadow_data_idx) & set(train_idx + non_member_idx)) == 0

    # Mix the groups
    train_idx = rng.permutation(train_idx)
    test_idx = rng.permutation(test_idx)
    non_member_idx = rng.permutation(non_member_idx)
    shadow_data_idx = rng.permutation(shadow_data_idx)
    if len(shadow_data_idx) < min_shadow_data:
        shadow_data_idx = np.array([], dtype=int)
    elif len(shadow_data_idx) % 2 == 1:
        shadow_data_idx = shadow_data_idx[:-1]  # Ensure we have an even number of samples, otherwise the shadow model attack raises an error

    split_dict = {
        'train_idx': train_idx,
        'test_idx': test_idx,
        'non_member_idx': non_member_idx,
        'shadow_train_idx': shadow_data_idx[:len(shadow_data_idx) // 2],
        'shadow_test_idx': shadow_data_idx[len(shadow_data_idx) // 2:]
    }
    if verbose:
        for key, val in split_dict.items():
            print(key, len(val), end=' | ')
        print("")

    return split_dict


def create_random_split_dict(run, split_dict):
    rng = np.random.default_rng(seed=run)
    ntrain = len(split_dict['train_idx'])
    nnmemb = len(split_dict['non_member_idx'])
    ntest = len(split_dict['test_idx'])
    idx_shuffled = rng.permutation(list(split_dict['train_idx']) + list(split_dict['non_member_idx']))
    split_dict_rand = split_dict.copy()
    split_dict_rand['train_idx'] = idx_shuffled[:ntrain]
    split_dict_rand['test_idx'] = idx_shuffled[ntrain:ntrain + ntest]
    split_dict_rand['non_member_idx'] = idx_shuffled[ntrain:ntrain + nnmemb]
    return split_dict_rand


def create_split_for_dataset(dataset, split_type, nsamples=-1, pval=-1, nruns=20):
    """
    Creates splits for the given processed dataset, using the subgroups information.
    We use a list to represent the number of samples from each group in the train/test/nonmemb sets; remaining samples are given as shadow data.
    Test samples are not relevant, we do not use them in the paper.
    - dataset: name of the dataset
    - split_type: determines how we assign samples to each group.
        - 'p_split': only two groups. Train is [int(pval*nsamples), nsamples-int(pval*nsamples)], non-memb is [nsamples/2, nsamples/2]
        - 'g0_split': only two groups. Train is [nsamples, 0], non-memb is [0, nsamples].
        - 'g1_split': only two groups. Train is [0, nsamples], non-memb is [nsamples, 0].
        - 'g_full_split': arbitrary number of groups. nsamples is not used. Let n0, n1, n2... be the number of samples per group. We generate:
            - train [n1, 0, 0, ...], non-memb [0, n2, n3, ...]  (no shadow data)
            - train [0, n2, 0, ...], non-memb [n1, 0, n2, ...]  (no shadow data)
            - etc.
    - nruns: number of splits to create for each dataset"""

    if nsamples == -1 and split_type != 'g_full_split':
        raise ValueError("Split type '{:s}' requires a number of samples.".format(split_type))
    if split_type == 'p_split' and pval == -1:
        raise ValueError("Split type 'p_split' requires a value of pval.")

    full_path_to_dataset = os.path.join(PROCESSED_DATASET_PATH, dataset + '.p')
    if not os.path.exists(full_path_to_dataset):
        raise ValueError("Processed dataset {:s} does not exist!".format(full_path_to_dataset))
    if not os.path.exists(SPLITS_PATH):
        os.mkdir(SPLITS_PATH)
    splits_dataset_subpath = os.path.join(SPLITS_PATH, dataset)
    if not os.path.exists(splits_dataset_subpath):
        os.mkdir(splits_dataset_subpath)
    with open(full_path_to_dataset, 'rb') as f:
        dataset_dict = pickle.load(f)

    subgroups = np.array(dataset_dict['subgroups'])
    ngroups = len(set(subgroups))
    idx_per_group = [list(np.where(subgroups == group)[0]) for group in range(ngroups)]
    if split_type == 'p_split':
        assert ngroups == 2
        ntrain_list = [[int(pval * nsamples), nsamples - int(pval * nsamples)]]
        ntest_list = [[int(pval * nsamples // 4), nsamples // 4 - int(pval * nsamples // 4)]]
        nnonmembers_list = [[nsamples // 2, nsamples - nsamples // 2]]
        prefix_list = ['n{:d}_p{:.2f}'.format(nsamples, pval)]
    elif split_type == 'g0_split':
        assert ngroups == 2
        ntrain_list = [[nsamples, 0]]
        ntest_list = [[nsamples // 4, 0]]
        nnonmembers_list = [[0, nsamples]]
        prefix_list = ['n{:d}_g0'.format(nsamples, pval)]
    elif split_type == 'g1_split':
        assert ngroups == 2
        ntrain_list = [[0, nsamples]]
        ntest_list = [[0, nsamples // 4]]
        nnonmembers_list = [[nsamples, 0]]
        prefix_list = ['n{:d}_g1'.format(nsamples, pval)]
    elif split_type == 'g_full_split':
        assert ngroups > 1
        ntrain_list = []
        ntest_list = []
        nnonmembers_list = []
        prefix_list = []
        group_sizes = [len(val) for val in idx_per_group]
        for group in range(ngroups):
            ntrain = [0] * ngroups
            ntrain[group] = group_sizes[group]
            nnonmembers = group_sizes[:]
            nnonmembers[group] = 0

            ntrain_list.append(ntrain)
            ntest_list.append(nnonmembers[:])
            nnonmembers_list.append(nnonmembers[:])
            prefix_list.append('nf{:d}_g{:d}'.format(group_sizes[group], group))
    else:
        raise ValueError("Split type '{:s}' not recognized".format(split_type))

    skip_count = 0
    files_created_count = 0
    for ntrain, ntest, nnonmemb, prefix in zip(ntrain_list, ntest_list, nnonmembers_list, prefix_list):

        splits_full_subpath = os.path.join(splits_dataset_subpath, prefix)
        splits_full_subpath_rand = os.path.join(splits_dataset_subpath, prefix + '_rand')
        if not os.path.exists(splits_full_subpath):
            os.mkdir(splits_full_subpath)
        if not os.path.exists(splits_full_subpath_rand):
            os.mkdir(splits_full_subpath_rand)
        for i_run in range(nruns):
            filename = prefix + '_{:02d}.p'.format(i_run)
            full_path_to_split_file = os.path.join(splits_full_subpath, filename)
            if os.path.exists(full_path_to_split_file):
                skip_count += 1
                with open(full_path_to_split_file, 'rb') as f:
                    split_dict = pickle.load(f)
            else:
                split_dict = create_split_dict(i_run, ntrain, ntest, nnonmemb, idx_per_group, nsamples)
                with open(full_path_to_split_file, 'wb') as f:
                    pickle.dump(split_dict, f)
                files_created_count += 1

            # Random split
            filename_rand = prefix + '_rand_{:02d}.p'.format(i_run)
            full_path_to_split_file_rand = os.path.join(splits_full_subpath_rand, filename_rand)

            if not os.path.exists(full_path_to_split_file_rand):
                split_dict_rand = create_random_split_dict(i_run, split_dict)
                with open(full_path_to_split_file_rand, 'wb') as f:
                    pickle.dump(split_dict_rand, f)
                files_created_count += 1

    if files_created_count == 0:
        print("CREATE_SPLITS: all splits for {:s} existed ({:d})".format(dataset, skip_count))
    else:
        print("CREATE_SPLITS: Done creating splits for {:s}: created {:d}, skipped {:d}".format(dataset, files_created_count, skip_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('type', type=str)
    parser.add_argument('--nsamples', type=int, default=-1)
    parser.add_argument('--nruns', type=int, default=20)
    parser.add_argument('--pval', type=float, default=0)
    args = parser.parse_args()
    print(args)

    create_split_for_dataset(args.dataset, args.type, nsamples=args.nsamples, pval=args.pval, nruns=args.nruns)


if __name__ == '__main__':
    main()
