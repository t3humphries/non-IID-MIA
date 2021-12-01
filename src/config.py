# GLOBAL VARIABLES
ORIGINAL_DATASET_PATH = './datasets_original'
PROCESSED_DATASET_PATH = './datasets'
SPLITS_PATH = './splits'
RESULTS_PATH = './results'
PLOTS_FOLDER = './plots'
MODEL = 'nn_'
PERTURBATION = 'grad_pert_'
DATASET_TO_N = {
    'adult-edu': 10_000,
    'adult-numedu': 10_000,
    'adult-edu-inv': 10_000,
    'adult-sex': 10_000,
    'adult-sex-inv': 10_000,
    'adult-clus2': 10_000,
    'adult-clus2-0': 10_000,
    'compas-clus2': 1_700,
    'compas-aa': 1_700,
    'compas-aa-inv': 1_700,
    'mary2': 1_000
}

DEFAULT_EPS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
P_LIST_FOR_SPLITS = [0., 0.5, 1.]
P_LIST_FOR_EVAL = P_LIST_FOR_SPLITS
SMALL_VALUE = 1e-16