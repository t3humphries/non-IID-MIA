from matplotlib import pyplot as plt
import pickle as pkl
import os
import numpy as np
import scipy.stats
from sklearn.metrics import roc_curve, roc_auc_score
from config import *
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 11})


def mean_confidence_interval(data, confidence=0.95):
    if data[0] is None:
        return np.nan, np.nan, np.nan
    else:
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
        return m, m - h, m + h


def get_advantage_bounds(epsilon_list, delta):
    if not isinstance(epsilon_list, (list, tuple, np.ndarray)):
        raise ValueError("epsilon_list should be a list, tuple, or numpy array")
    if isinstance(epsilon_list, (list, tuple)):
        epsilon_list = np.array(epsilon_list)

    yeom_bound = np.exp(np.minimum(epsilon_list, np.log(2))) - 1
    erl_bound = 1 - np.exp(-epsilon_list) * (1 - delta)
    our_bound = (1 - np.exp(-epsilon_list) + 2 * delta) / (1 + np.exp(-epsilon_list))
    return yeom_bound, erl_bound, our_bound


def load_results(dataset_name, split_name):

    results_path = os.path.join(RESULTS_PATH, dataset_name, split_name)
    prefix = '{:s}_{:s}_'.format(dataset_name, split_name)
    suffix = '.p'
    results = {}
    count = 0
    if os.path.exists(results_path):
        for file in os.scandir(results_path):
            if file.name.startswith(prefix) and file.name.endswith(suffix):
                eps, run = file.name[len(prefix):-len(suffix)].split('_')
                eps = float(eps) if not eps == 'nan' else 1e4
                run = int(run)
                if eps not in results:
                    results[eps] = {}
                with open(os.path.join(results_path, file.name), 'rb') as f:
                    try:
                        results[eps][run] = pkl.load(f)
                    except EOFError:
                        print('BAD FILE SKIPPING:',os.path.join(results_path, file.name))
                count += 1
    print("  Loaded {:d} files for {:s} from {:s}".format(count, dataset_name, results_path))
    return results


def process_loaded_results(results):
    """Returns processed_results, which is a dictionary where:
    processed_results[epsilon][avg_acc_train] = list (length=nruns)
    processed_results[epsilon][avg_acc_validation] = list (length=nruns)
    processed_results[epsilon][avg_acc_non_mem] = list (length=nruns)
    processed_results[epsilon][shadow_attack_adv] = list (length=nruns)
    processed_results[epsilon][log_attack_adv] = list (length=nruns)
    processed_results[epsilon][optlog_attack_adv] = list (length=nruns)
    """

    def perform_optimal_logloss_attack(results, eps_list):
        """Optimal logloss attack"""

        avg_adv = []
        adv_per_eps = {}
        decisions = {}
        for eps in eps_list:
            decisions[eps] = {}
            adv_per_eps[eps] = []
            if eps not in results:
                avg_adv.append(np.nan)
            else:
                adv = []
                for run in results[eps].keys():
                    memb = results[eps][run]['membership_indicator']
                    conf_int = results[eps][run]['target_predictions']
                    n = conf_int.shape[0]
                    true_labels = results[eps][run]['true_labels']
                    chosen_conf = np.array(conf_int[range(n), true_labels])[0]
                    log_losses = -np.log(np.maximum(chosen_conf, 1e-16))

                    fpr, tpr, thresholds = roc_curve(memb, -log_losses, pos_label=1)

                    chosen_threshold = thresholds[np.argmax(tpr - fpr)]
                    decisions[eps][run] = 1 * (log_losses <= -chosen_threshold)

                    # Debug: this should be the same as the advantage
                    tpr_from_decisions = np.mean(decisions[eps][run][memb == 1] == 1)
                    fpr_from_decisions = np.mean(decisions[eps][run][memb == 0] == 1)
                    if np.abs(np.max(tpr - fpr) - (tpr_from_decisions - fpr_from_decisions)) > 1e-9:
                        raise ValueError("OPT LL: Advantage: ", np.max(tpr - fpr), " From decision: ", tpr_from_decisions - fpr_from_decisions)

                    adv.append(np.max(tpr - fpr))
                    adv_per_eps[eps].append(np.max(tpr - fpr))
                avg_adv.append(np.mean(adv))
        return avg_adv, decisions, adv_per_eps


    key_list = ['avg_acc_train', 'avg_acc_validation', 'avg_acc_non_mem', 'shadow_attack_adv', 'log_attack_adv']
    processed_results = {}
    for eps in sorted(results.keys()):
        processed_results_this_eps = {key: [] for key in key_list}
        for run in results[eps].keys():
            for key in key_list:
                processed_results_this_eps[key].append(results[eps][run][key])
        processed_results[eps] = processed_results_this_eps
    # Add optimal logloss attack:
    _, _, adv_per_eps = perform_optimal_logloss_attack(results, results.keys())
    for eps in results.keys():
        processed_results[eps]['optlog_attack_adv'] = adv_per_eps[eps]
    return processed_results


def plot_advantage_in_axis(ax, result_list, color_list, label_list, attack_adv_name):
    for result, color, label in zip(result_list, color_list, label_list):
        eps_list = list(sorted(result.keys()))
        avg, hi, lo = zip(*[mean_confidence_interval(result[eps][attack_adv_name]) for eps in eps_list])
        ax.semilogx(eps_list, avg, '.-', color=color, label=label)
        ax.fill_between(eps_list, lo, hi, color=color, alpha=0.3)
    ylim_beforebounds = ax.get_ylim()
    xlim = ax.get_xlim()
    xvals = np.logspace(np.log10(xlim[0]), np.log10(xlim[-1]), 100)
    _, _, bound = get_advantage_bounds(xvals, 1e-5)
    ax.semilogx(xvals, bound, '--', color='k', alpha=0.5, label='IID bound')
    ax.semilogx(xvals, xvals * 0, '-', color='k', alpha=0.3)
    ax.set_ylabel('Membership Advantage')
    ax.set_xlabel('$\epsilon$')
    ax.set_xlim([0.8e-2, 2e4])
    ax.set_xticks([1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    ax.set_xticklabels(['$0$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$', '$\infty$'])
    return ylim_beforebounds


def plot_accuracy_in_axis(ax, result_list, color_list, label_list, plot_validation = False):
    if plot_validation:
        styles = ['-', '--', ':']
        mstyles = ['.', 'x', '*']
        labels_acc = ['members', 'non-members', 'validation']
        keys = ['avg_acc_train', 'avg_acc_non_mem', 'avg_acc_validation']
    else:
        styles = ['-', '--']
        mstyles = ['.', 'x']
        labels_acc = ['members', 'non-members']
        keys = ['avg_acc_train', 'avg_acc_non_mem']

    for result, color, label in zip(result_list, color_list, label_list):
        eps_list = list(sorted(result.keys()))
        for i, key in enumerate(keys):
            avg, hi, lo = zip(*[mean_confidence_interval(result[eps][key]) for eps in eps_list])
            ax.semilogx(eps_list, avg, linestyle=styles[i], marker=mstyles[i], color=color, label=label)
            ax.fill_between(eps_list, lo, hi, color=color, alpha=0.3)
    ax.legend([Line2D([0], [0], color='k', linewidth=2, linestyle=sty, marker=mark) for mark, sty in zip(mstyles, styles)],
              labels_acc)
    ylim_beforebounds = ax.get_ylim()
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('$\epsilon$')
    ax.set_title('Classification Accuracy')
    ax.set_xticks([1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4])
    ax.set_xticklabels(['$0$', '$10^{-1}$', '$10^{0}$', '$10^{1}$', '$10^{2}$', '$10^{3}$', '$\infty$'])
    return ylim_beforebounds


def plot_adult_cluster(savefig=False):

    for dataset_name, split_name in [('adult-clus2-0', 'n10000_g0'), ('adult-clus2-0', 'n10000_g1')]:
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))

        results = process_loaded_results(load_results(dataset_name, split_name))
        results_rand = process_loaded_results(load_results(dataset_name, split_name + '_rand'))
        results_list = [results, results_rand]
        color_list = ['C0', 'C1']
        label_list = ['non-IID', 'IID']
        ylim_list = []
        for i_att, att in enumerate(['log_attack_adv', 'optlog_attack_adv', 'shadow_attack_adv']):
            ylim_beforebounds = plot_advantage_in_axis(ax[i_att], results_list, color_list, label_list, att)
            ylim_list.append(ylim_beforebounds)
        chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
        chosen_ylim_min = np.min(np.min([ylim[0] for ylim in ylim_list]), 0)
        for i_att in range(3):
            ax[i_att].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
        ax[0].set_title("Average Threshold Attack")
        ax[1].set_title("Optimal Threshold Attack")
        ax[2].set_title("Shadow Model Attack")

        # Plot model's accuracy
        plot_accuracy_in_axis(ax[3], results_list, color_list, label_list)
        ax[3].set_title("Model's Accuracy")
        ax[0].legend()
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}.pdf'.format(dataset_name)))
        plt.suptitle(dataset_name)
        plt.tight_layout()
        plt.show()


def plot_compas_cluster(savefig=False):

    for dataset_name, split_name in [('compas-clus2-2', 'n2000_g0')]: #, ('compas-clus2-2', 'n2000_g1')]:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))

        results = process_loaded_results(load_results(dataset_name, split_name))
        results_rand = process_loaded_results(load_results(dataset_name, split_name + '_rand'))
        results_list = [results, results_rand]
        color_list = ['C0', 'C1']
        label_list = ['non-IID', 'IID']
        ylim_list = []
        ylim_beforebounds = plot_advantage_in_axis(ax[0], results_list, color_list, label_list, 'optlog_attack_adv')
        ylim_list.append(ylim_beforebounds)
        chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
        chosen_ylim_min = np.min(np.min([ylim[0] for ylim in ylim_list]), 0)
        ax[0].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
        ax[0].set_title("Optimal Threshold Attack")
        ax[0].legend()

        # Plot model's accuracy
        plot_accuracy_in_axis(ax[1], results_list, color_list, label_list)
        ax[1].set_title("Model's Accuracy")
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}_{:s}.pdf'.format(dataset_name, split_name)))
        plt.suptitle(dataset_name)
        plt.tight_layout()
        plt.show()


def plot_attribute(savefig=False, short=False):

    pval_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    if not short:
        attack_list = ['log_attack_adv', 'optlog_attack_adv', 'shadow_attack_adv']
        attack_names = ["Average Threshold Attack", "Optimal Threshold Attack", "Shadow Model Attack"]
        file_suffix = ''
        legend_pos = 1
        exp_list = [('adult-sex', 10_000)]
    else:
        attack_list = ['optlog_attack_adv']
        attack_names = ['Optimal Threshold Attack']

        file_suffix = '_short'
        legend_pos = 0
        exp_list = [('adult-numedu', 10_000)]
    for dataset_name, nsamples in exp_list:
        fig, ax = plt.subplots(1, len(attack_list) + 1, figsize=(4 * (len(attack_list) + 1), 3.5))
        results_list = []
        color_list = []
        label_list = []
        for i_pval, pval in enumerate(pval_list):
            results = process_loaded_results(load_results(dataset_name, 'n{:d}_p{:.2f}'.format(nsamples, pval)))
            results_list.append(results)
            color_list.append('C{:d}'.format(i_pval))
            label_list.append('p={:.2f}'.format(pval))
        ylim_list = []
        for i_att, att in enumerate(attack_list):
            ylim_beforebounds = plot_advantage_in_axis(ax[i_att], results_list, color_list, label_list, att)
            ylim_list.append(ylim_beforebounds)
        chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
        chosen_ylim_min = min(np.min([ylim[0] for ylim in ylim_list]), -0.10)
        for i_att in range(len(attack_list)):
            ax[i_att].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
            ax[i_att].set_title(attack_names[i_att])
        ax[legend_pos].legend(ncol=2)

        idx = [0, 4]
        plot_accuracy_in_axis(ax[len(attack_list)], [results_list[i] for i in idx], [color_list[i] for i in idx], [label_list[i] for i in idx], plot_validation=True)
        ax[len(attack_list)].set_title("Model's Accuracy")
        styles = ['-', '--', ':']
        mstyles = ['.', 'x', '*']
        labels_acc = ['members', 'non-members', 'validation']
        leg1 = ax[len(attack_list)].legend([Line2D([0], [0], color='k', linewidth=2, linestyle=sty, marker=mark) for mark, sty in zip(mstyles, styles)],
                  labels_acc, loc='lower right')
        leg2 = ax[len(attack_list)].legend([Line2D([0], [0], color=color, linewidth=2, linestyle='', marker='o') for color in [color_list[i] for i in idx]],
                  [label_list[i] for i in idx], loc='upper left')
        ax[len(attack_list)].add_artist(leg1)

        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}{:s}.pdf'.format(dataset_name, file_suffix)))
        plt.tight_layout()
        plt.show()


def plot_heart(savefig=False):

    dataset_name = 'heart-mix'
    nsamples_list = (303, 294, 123, 200)
    names_list = ['Cleveland', 'Hungary', 'Switzerland', 'VA Long Beach']
    fig, ax = plt.subplots(1, len(nsamples_list), figsize=(4 * len(nsamples_list), 3.5))
    ylim_list = []
    for group, nsamples in enumerate(nsamples_list):
        results = process_loaded_results(load_results(dataset_name, 'nf{:d}_g{:d}'.format(nsamples, group)))
        results_rand = process_loaded_results(load_results(dataset_name, 'nf{:d}_g{:d}_rand'.format(nsamples, group)))
        results_list = [results, results_rand]
        color_list = ['C0', 'C1']
        label_list = ['non-IID', 'IID']

        att = 'optlog_attack_adv'
        ylim_beforebounds = plot_advantage_in_axis(ax[group], results_list, color_list, label_list, att)
        ylim_list.append(ylim_beforebounds)
    chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
    chosen_ylim_min = np.min(np.min([ylim[0] for ylim in ylim_list]), 0)
    for group in range(len(nsamples_list)):
        ax[group].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
        ax[group].set_title(names_list[group])

    ax[1].legend()
    if savefig:
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}.pdf'.format(dataset_name)))
    plt.suptitle(dataset_name)
    plt.tight_layout()
    plt.show()


def plot_students_short(savefig=False):

    for dataset_name, split_name in [('students2', 'nf423_g0')]:
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))

        results = process_loaded_results(load_results(dataset_name, split_name))
        results_rand = process_loaded_results(load_results(dataset_name, split_name + '_rand'))
        results_list = [results, results_rand]
        color_list = ['C0', 'C1']
        label_list = ['non-IID', 'IID']
        ylim_list = []
        ylim_beforebounds = plot_advantage_in_axis(ax[0], results_list, color_list, label_list, 'optlog_attack_adv')
        ylim_list.append(ylim_beforebounds)
        chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
        chosen_ylim_min = np.min(np.min([ylim[0] for ylim in ylim_list]), 0)
        ax[0].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
        ax[0].set_title("Optimal Threshold Attack")
        ax[0].legend()

        # Plot model's accuracy
        plot_accuracy_in_axis(ax[1], results_list, color_list, label_list)
        ax[1].set_title("Model's Accuracy")
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}_{:s}.pdf'.format(dataset_name, split_name)))
        plt.suptitle(dataset_name)
        plt.tight_layout()
        plt.show()


def plot_full_group_dataset_split(savefig=False):

    for dataset_name, nsamples_list in [('heart-mix', (303, 294, 123, 200)), ('students2', (423, 226))]:
        fig, ax = plt.subplots(3, len(nsamples_list), figsize=(4 * len(nsamples_list), 3 * 3))
        for group, nsamples in enumerate(nsamples_list):
            results = process_loaded_results(load_results(dataset_name, 'nf{:d}_g{:d}'.format(nsamples, group)))
            results_rand = process_loaded_results(load_results(dataset_name, 'nf{:d}_g{:d}_rand'.format(nsamples, group)))
            results_list = [results, results_rand]
            color_list = ['C0', 'C1']
            label_list = ['non-IID', 'IID']
            ylim_list = []
            for i_att, att in enumerate(['log_attack_adv', 'optlog_attack_adv']):
                ylim_beforebounds = plot_advantage_in_axis(ax[i_att, group], results_list, color_list, label_list, att)
                ylim_list.append(ylim_beforebounds)
            chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
            chosen_ylim_min = np.min(np.min([ylim[0] for ylim in ylim_list]), 0)
            for i_att in range(2):
                ax[i_att, group].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
            ax[0, group].set_title("Average Threshold Attack")
            ax[1, group].set_title("Optimal Threshold Attack")

            # Plot model's accuracy
            plot_accuracy_in_axis(ax[2, group], results_list, color_list, label_list)
            ax[2, group].set_title("Model's Accuracy")
        ax[0, 0].legend()
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}.pdf'.format(dataset_name)))
        plt.suptitle(dataset_name)
        plt.tight_layout()
        plt.show()


def plot_full_group_dataset_split_small(savefig=False):

    for dataset_name, split_name in [('heart-mix', 'nf294_g1'), ('heart-mix', 'nf123_g2'), ('students2', 'nf423_g0'),
                                     ('compas-clus2-2', 'n2000_g0'), ('compas-clus2-2', 'n2000_g1')]:
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))

        results = process_loaded_results(load_results(dataset_name, split_name))
        results_rand = process_loaded_results(load_results(dataset_name, split_name + '_rand'))
        results_list = [results, results_rand]
        color_list = ['C0', 'C1']
        label_list = ['non-IID', 'IID']
        ylim_list = []
        for i_att, att in enumerate(['log_attack_adv', 'optlog_attack_adv']):
            ylim_beforebounds = plot_advantage_in_axis(ax[i_att], results_list, color_list, label_list, att)
            ylim_list.append(ylim_beforebounds)
        chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
        chosen_ylim_min = np.min(np.min([ylim[0] for ylim in ylim_list]), 0)
        for i_att in range(2):
            ax[i_att].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
        ax[0].set_title("Average Threshold Attack")
        ax[1].set_title("Optimal Threshold Attack")

        # Plot model's accuracy
        plot_accuracy_in_axis(ax[2], results_list, color_list, label_list)
        ax[2].set_title("Model's Accuracy")
        ax[0].legend()
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}.pdf'.format(dataset_name)))
        plt.suptitle(dataset_name)
        plt.tight_layout()
        plt.show()


def plot_group_dataset_split(savefig=False):

    for dataset_name, split_name in [('kdd-census2', 'n10000_g0'), ('texas-region-3', 'n10000_g0')]:
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.5))

        results = process_loaded_results(load_results(dataset_name, split_name))
        results_rand = process_loaded_results(load_results(dataset_name, split_name + '_rand'))
        results_list = [results, results_rand]
        color_list = ['C0', 'C1']
        label_list = ['non-IID', 'IID']
        ylim_list = []
        for i_att, att in enumerate(['log_attack_adv', 'optlog_attack_adv', 'shadow_attack_adv']):
            ylim_beforebounds = plot_advantage_in_axis(ax[i_att], results_list, color_list, label_list, att)
            ylim_list.append(ylim_beforebounds)
        chosen_ylim_max = np.max([ylim[1] for ylim in ylim_list])
        chosen_ylim_min = np.min(np.min([ylim[0] for ylim in ylim_list]), 0)
        for i_att in range(3):
            ax[i_att].set_ylim([chosen_ylim_min, 1.05 * chosen_ylim_max])
        ax[0].set_title("Average Threshold Attack")
        ax[1].set_title("Optimal Threshold Attack")
        ax[2].set_title("Shadow Model Attack")

        # Plot model's accuracy
        plot_accuracy_in_axis(ax[3], results_list, color_list, label_list)
        ax[3].set_title("Model's Accuracy")
        ax[0].legend()
        if savefig:
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER,'results_{:s}.pdf'.format(dataset_name)))
        plt.suptitle(dataset_name)
        plt.tight_layout()
        plt.show()


def plot_bound_comparison():

    epsilon_list = np.logspace(-2, 1, 100)
    delta = 1e-5
    _, erl, ours = get_advantage_bounds(epsilon_list, delta)
    yeom = np.exp(epsilon_list) - 1

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.semilogx(epsilon_list, yeom, ':', label='Yeom et al. (Eq. 6)')
    ax.semilogx(epsilon_list, erl, '--', label='Erlingsson et al. (Eq. 7)')
    ax.semilogx(epsilon_list, ours, '-', label='Tighter bound (Eq. 8)')
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.set_xlabel("$\\epsilon$")
    ax.set_ylabel("Membership Advantage")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_FOLDER, 'bound_comparison.pdf'))
    plt.show()

def main():
    plot_adult_cluster(savefig=True)
    plot_compas_cluster(savefig=True)
    plot_attribute(savefig=True, short=False)
    plot_attribute(savefig=True, short=True)
    plot_heart(savefig=True)
    plot_students_short(savefig=True)
    plot_group_dataset_split(savefig=True)

    # plot_full_group_dataset_split(savefig=False)
    # plot_full_group_dataset_split_small(savefig=True)
    # plot_bound_comparison()

if __name__ == "__main__":
    main()