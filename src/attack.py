try:
    import mkl

    mkl.set_num_threads(1)
    print(mkl.get_max_threads())
except ImportError:
    pass
import sys
from output_filter import OutputFilter

sys.stdout = OutputFilter(sys.stdout)
sys.stderr = OutputFilter(sys.stderr)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.compat.v1 as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.disable_v2_behavior()

from classifier import train as train_model, train_private, iterate_minibatches, get_predictions
from sklearn.metrics import roc_curve
import numpy as np
import theano
import theano.tensor as T
import lasagne
import argparse
import os
import pickle
import multiprocessing
import copy
import time
from compute_noise import update_noise_multipliers
from config import *
from create_splits import parse_split_name

def train_target_model(dataset, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn', privacy='no_privacy', dp='dp',
                       epsilon=0.5, delta=1e-5, non_linearity='relu', C=1):
    train_x, train_y, test_x, test_y, non_owner_x, non_owner_y = dataset

    classifier, train_loss, train_acc, test_acc, non_owner_acc = train_private(dataset, n_hidden=n_hidden, epochs=epochs,
                                                                               learning_rate=learning_rate, batch_size=batch_size, model=model, l2_ratio=l2_ratio,
                                                                               silent=False, privacy=privacy, dp=dp, epsilon=epsilon, delta=delta,
                                                                               non_linearity=non_linearity, C=C)
    # test data for attack model
    attack_x, attack_y = [], []

    # data used in training, label is 1
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.concatenate((train_x, non_owner_x))},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)

    attack_y.append(np.ones(train_x.shape[0]))
    attack_y.append(np.zeros(non_owner_x.shape[0]))

    attack_x = pred_scores.astype('float32')
    attack_y = np.concatenate(attack_y).astype('int32')

    classes = np.concatenate([train_y, non_owner_y])
    return attack_x, attack_y, classes, train_loss, classifier, train_acc, test_acc, non_owner_acc


def train_shadow_models(split_data, n_hidden=50, epochs=100, n_shadow=20, learning_rate=0.05, batch_size=100,
                        l2_ratio=1e-7, model='nn'):
    # for getting probabilities
    input_var = T.matrix('x')
    # for attack model
    attack_x, attack_y = [], []
    classes = []

    train_x = split_data['shadow_train_x']
    train_y = split_data['shadow_train_y']
    test_x = split_data['shadow_test_x']
    test_y = split_data['shadow_test_y']

    assert len(train_x) == len(test_x)
    shadow_indices = list(range(len(train_x)))
    size_shadow_datasets = int(2 * len(train_x) / n_shadow)
    for i in range(n_shadow):
        print('\nTraining shadow model {}'.format(i))
        shadow_i_indices = np.random.choice(shadow_indices, size_shadow_datasets, replace=False)
        shadow_data = train_x[shadow_i_indices], train_y[shadow_i_indices], test_x[shadow_i_indices], test_y[shadow_i_indices]

        # train model
        output_layer, _, _, _, _, _ = train_model(shadow_data, n_hidden=n_hidden, epochs=epochs,
                                                  learning_rate=learning_rate, batch_size=batch_size, model=model, l2_ratio=l2_ratio,
                                                  silent=False)
        prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)
        prob_fn = theano.function([input_var], prob, allow_input_downcast=True)
        # print('Gather training data for attack model')
        attack_i_x, attack_i_y = [], []

        shadow_train_x, shadow_train_y, shadow_test_x, shadow_test_y = shadow_data
        # data used in training, label is 1
        for batch in iterate_minibatches(shadow_train_x, shadow_train_y, batch_size, False):
            attack_i_x.append(prob_fn(batch[0]))
            attack_i_y.append(np.ones(batch_size))
        # data not used in training, label is 0
        for batch in iterate_minibatches(shadow_test_x, shadow_test_y, batch_size, False):
            attack_i_x.append(prob_fn(batch[0]))
            attack_i_y.append(np.zeros(batch_size))
        attack_x += attack_i_x
        attack_y += attack_i_y
        classes.append(np.concatenate([shadow_train_y, shadow_test_y]))
    # train data for attack model
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate(classes)

    return attack_x, attack_y, classes


def train_attack_model(classes, dataset=None, n_hidden=50, learning_rate=0.01, batch_size=200, epochs=50,
                       model='nn', l2_ratio=1e-7):
    train_x, train_y, test_x, test_y = dataset

    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    true_y = []
    pred_y = []
    pred_scores = np.empty(len(test_y))
    true_x = []
    for c in unique_classes:
        print('\nTraining attack model for class {}...'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        _, c_pred_y, c_pred_scores, _, _, _ = train_model(c_dataset, n_hidden=n_hidden, epochs=epochs,
                                                          learning_rate=learning_rate, batch_size=batch_size, model=model, l2_ratio=l2_ratio,
                                                          non_linearity='relu', silent=False)
        true_y.append(c_test_y)
        pred_y.append(c_pred_y)
        true_x.append(c_test_x)
        # place c_pred_scores where it belongs in pred_scores (train, then test)
        pred_scores[c_test_indices] = c_pred_scores[:, 1]

    print('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    true_x = np.concatenate(true_x)

    fpr, tpr, thresholds = roc_curve(true_y, pred_y, pos_label=1)
    # print(fpr, tpr, tpr - fpr)
    attack_adv = tpr[1] - fpr[1]
    print('Done Shadow Model attack, advantage: {:.5f} (advantage without ROC formula {:.5f})'.format(attack_adv, 1 - 2 * np.mean(np.abs(true_y - pred_y))))

    return attack_adv, pred_scores


def load_data(dataset_name, split_name, run):
    with open(os.path.join(PROCESSED_DATASET_PATH, '{:s}.p'.format(dataset_name)), 'rb') as f:
        dataset = pickle.load(f)

    split_file_path = os.path.join(SPLITS_PATH, dataset_name, split_name, '{:s}_{:02d}.p'.format(split_name, run))
    with open(split_file_path, 'rb') as f:
        split = pickle.load(f)

    split_data = {
        'train_x': dataset['features'][split['train_idx']],
        'train_y': dataset['labels'][split['train_idx']],

        'test_x': dataset['features'][split['test_idx']],
        'test_y': dataset['labels'][split['test_idx']],

        'non_member_x': dataset['features'][split['non_member_idx']],
        'non_member_y': dataset['labels'][split['non_member_idx']],

        'shadow_train_x': dataset['features'][split['shadow_train_idx']],
        'shadow_train_y': dataset['labels'][split['shadow_train_idx']],

        'shadow_test_x': dataset['features'][split['shadow_test_idx']],
        'shadow_test_y': dataset['labels'][split['shadow_test_idx']]
    }
    return split_data


def shadow_model_attack(args, attack_test_x, attack_test_y, test_classes, split_data):
    print('-' * 10 + 'BEGIN SHADOW MODEL ATTACK' + '-' * 10 + '\n', flush=True)

    print('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n', flush=True)
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        split_data,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
    )

    print('\n' + '-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n', flush=True)
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    return train_attack_model(
        dataset=dataset,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        n_hidden=args.attack_n_hidden,
        l2_ratio=args.attack_l2_ratio,
        model=args.attack_model,
        classes=(train_classes, test_classes),
    )


def log_loss_attack(true_y, pred_y, membership, train_loss):
    print('-' * 10 + 'BEGIN LOG LOSS ATTACK' + '-' * 10 + '\n', flush=True)
    # The actual attack
    pred_membership = np.where(log_loss(true_y, pred_y) <= train_loss, 1, 0)

    # Calculating the results
    fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
    mem_adv = tpr[1] - fpr[1]

    print('Done Log Loss attack, advantage: {:.5f} (advantage without ROC formula {:.5f}'.format(mem_adv, 1 - 2 * np.mean(np.abs(np.array(membership) - pred_membership))))

    return mem_adv, pred_membership, log_loss(true_y, pred_y)


def log_loss(a, b):
    return [-np.log(max(b[i, a[i]], SMALL_VALUE)) for i in range(len(a))]


def run_experiment(args):

    t_init = time.time()

    if args.target_privacy == 'no_privacy':
        filename = args.dataset_name + '_' + args.split_name + '_' + '{:.2f}_{:02d}.p'.format(np.NaN, args.run)
    else:
        filename = args.dataset_name + '_' + args.split_name + '_' + '{:.2f}_{:02d}.p'.format(args.target_epsilon, args.run)

    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        RESULTS_PATH,
        args.dataset_name,
        args.split_name,
        filename
    )

    if os.path.exists(save_path):
        print('* {:s} Data already collected at save path: {:s}'.format(filename, save_path))
        return

    # print('-' * 10 + 'TRAIN TARGET (RUN ' + str(args.run) + ')' + '-' * 10 + '\n', flush=True)
    print("* {:s} Going to train target model".format(filename))
    split_data = load_data(args.dataset_name, args.split_name, args.run)
    dataset = (
        split_data['train_x'],
        split_data['train_y'],
        split_data['test_x'],
        split_data['test_y'],
        split_data['non_member_x'],
        split_data['non_member_y']
    )

    pred_y, membership, true_y, train_loss, classifier, train_acc, test_acc, non_owner_acc = train_target_model(
        dataset=dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        privacy=args.target_privacy,
        dp=args.target_dp,
        epsilon=args.target_epsilon,
        delta=args.target_delta,
        C=args.clipping_threshold
    )

    if len(split_data['shadow_train_x']) > 0 and not args.noshadow:
        print("* {:s} Done training, running shadow model attack ({:.1f} secs)".format(filename, time.time() - t_init))
        shadow_attack_adv, shadow_attack_pred = shadow_model_attack(args, pred_y, membership, true_y, split_data)
        print("* {:s} Shadow model attack advantage = {:.3f} starting logloss attack ({:.1f} secs)".format(filename, shadow_attack_adv, time.time() - t_init))
    else:
        print("* {:s} Done training, skipping shadow model attack ({:.1f} secs)".format(filename, time.time() - t_init))
        shadow_attack_adv = shadow_attack_pred = None

    log_attack_adv, log_attack_pred, log_losses = log_loss_attack(true_y, pred_y, membership, train_loss)
    print("* {:s} Logloss attack advantage = {:.3f} saving and exiting ({:.1f} secs)".format(filename, log_attack_adv, time.time() - t_init))
    # print('-' * 10 + 'SAVING RESULTS' + '-' * 10, flush=True)

    output_dictionary = {
        'avg_acc_train': train_acc,
        'avg_acc_validation': test_acc,
        'avg_acc_non_mem': non_owner_acc,
        'avg_log_loss_target': train_loss,
        'shadow_attack_adv': shadow_attack_adv,
        'shadow_attack_pred': shadow_attack_pred,
        'log_attack_adv': log_attack_adv,
        'log_attack_pred': log_attack_pred,
        'membership_indicator': membership,
        'true_labels': true_y,
        'target_predictions': pred_y
    }
    with open(save_path, 'wb') as f:
        pickle.dump(output_dictionary, f)


def main():

    print("-***- Starting!")
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', type=str)
    # parser.add_argument('--split_name', type=str, default='')
    parser.add_argument('--nmaxruns', type=int, default=100)
    # parser.add_argument('--pvals', type=str, default='')
    parser.add_argument('--noshadow', default=False, action='store_true')

    # Dataset options
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--class_size', type=int, default=0)

    # Target model configuration
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-5)
    parser.add_argument('--target_privacy', type=str, default='grad_pert')
    parser.add_argument('--target_dp', type=str, default='rdp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    # parser.add_argument('--target_epsilons', type=float, nargs='*', default=DEFAULT_EPS)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--clipping_threshold', type=int, default=1)

    # Attack configuration
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)
    parser.add_argument('--n_shadow', type=int, default=5)

    # parse configuration
    args = parser.parse_args()

    t0 = time.time()
    # jobs = []
    print("-***- ATTACK: Starting main for dataset {:s}".format(args.dataset_name))
    if not os.path.exists(os.path.join(PROCESSED_DATASET_PATH, args.dataset_name + '.p')):
        raise ValueError("Dataset {:s} does not exist!".format(args.dataset_name))
    print("-***- Reading splits from folders ", end='')
    split_folder_names = [file.name for file in os.scandir(os.path.join(SPLITS_PATH, args.dataset_name))]
    print(" and found {:d} splits".format(len(split_folder_names)))

    if args.target_privacy == 'no_privacy':
        epsilon_list = [np.nan]
    else:
        ntrain_list = list(set([parse_split_name(file)[0] for file in split_folder_names]))
        print("ntrain_list: ", ntrain_list)
        for train_size in ntrain_list:
            print("-***- Updating noise multipliers for ntrain_size = {:d}".format(train_size))
            update_noise_multipliers(train_size, args.target_batch_size, args.target_epochs, args.target_delta, DEFAULT_EPS)
        epsilon_list = [np.nan] + DEFAULT_EPS

    count_skip = 0
    jobs_per_run = {}
    for split_folder in split_folder_names:
        run_list = [int(file.name.split("_")[-1][:-2]) for file in os.scandir(os.path.join(SPLITS_PATH, args.dataset_name, split_folder))]
        print("run_list: ", run_list)
        run_list = [run for run in run_list if run <= args.nmaxruns]
        if len(run_list) > 0:
            if not os.path.exists(os.path.join(RESULTS_PATH, args.dataset_name, split_folder)):
                os.makedirs(os.path.join(RESULTS_PATH, args.dataset_name, split_folder))  # Make the results folder if it doesn't exist!
            for epsilon in epsilon_list:
                for run in run_list:
                    results_filename = args.dataset_name + '_' + split_folder + '_' + '{:.2f}_{:02d}.p'.format(epsilon, run)
                    if os.path.exists(os.path.join(RESULTS_PATH, args.dataset_name, split_folder, results_filename)):
                        # print("Results {:s} found, skipping it...".format(results_filename))
                        count_skip += 1
                    else:
                        args_copy = copy.deepcopy(args)
                        args_copy.split_name = split_folder
                        if not np.isnan(epsilon):
                            args_copy.target_epsilon = epsilon
                            args_copy.target_privacy = 'grad_pert'
                        else:
                            args_copy.target_privacy = 'no_privacy'
                        args_copy.run = run
                        if run not in jobs_per_run:
                            jobs_per_run[run] = [args_copy]
                        else:
                            jobs_per_run[run].append(args_copy)
                        # jobs.append(args_copy)
    jobs = []  # New: jobs per run
    for run in sorted(jobs_per_run.keys()):
        jobs += jobs_per_run[run]

    # for i, job in enumerate(jobs):
    #     print("{:d}) split={:s}, {:s}, eps={:.2f}, run={:d}".format(i, job.split_name, job.target_privacy, job.target_epsilon, job.run))

    number_workers = multiprocessing.cpu_count() - 2
    print('-***-Initializing pool with {:d} workers, we have {:d} jobs ({:d} skipped)'.format(number_workers, len(jobs), count_skip))

    pool = multiprocessing.Pool(number_workers)
    pool.map(run_experiment, jobs, chunksize=1)

    print("-***-Done experiment, elapsed time {:.1f} seconds".format(time.time() - t0))


if __name__ == '__main__':

    main()
