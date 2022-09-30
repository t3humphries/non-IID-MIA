import numpy as np
import time
import os
import pickle
import dp_accounting
rdp_acc = dp_accounting.rdp.rdp_privacy_accountant


NOISE_MULT_FILE = 'noise_multipliers.p'


def update_noise_multipliers(train_size, batch_size, epochs, delta, epsilon_list):

    if not os.path.exists(NOISE_MULT_FILE):
        noise_multipliers = {}
    else:
        with open(NOISE_MULT_FILE, 'rb') as f:
            noise_multipliers = pickle.load(f)

    key = '{:d}_{:d}_{:d}_{:.1e}'.format(train_size, batch_size, epochs, delta)
    if key not in noise_multipliers:
        noise_multipliers[key] = {}
    params_dict = {
        'train_x_size0': train_size,
        'batch_size': batch_size,
        'epochs': epochs
    }
    change = False
    for epsilon in epsilon_list:
        if epsilon not in noise_multipliers[key]:
            change = True
            sigma = compute_sigma_given_epsilon_bisection(epsilon, delta, 1e-4, params_dict, verbose=True)
            noise_multipliers[key][epsilon] = sigma
    if change:
        with open(NOISE_MULT_FILE, 'wb') as f:
            pickle.dump(noise_multipliers, f)
        print("-***- Updated the noise multipliers table!")
    else:
        print("-***- Didn't need to update the noise multipliers table!")


def get_noise_multiplier(train_size, batch_size, epochs, delta):
    if not os.path.exists(NOISE_MULT_FILE):
        raise ValueError("Noise multipliers file {:s} does not exist!".format(NOISE_MULT_FILE))
    with open(NOISE_MULT_FILE, 'rb') as f:
        noise_multipliers = pickle.load(f)
    key = '{:d}_{:d}_{:d}_{:.1e}'.format(train_size, batch_size, epochs, delta)
    if key not in noise_multipliers:
        raise ValueError("Key {:s} does not exist in the noise multipliers file!".format(key))
    return noise_multipliers[key]


def compute_sigma_given_epsilon_bisection(target_epsilon, delta, tolerance, params, verbose=False):
    v_print = print if verbose else lambda *a, **k: None

    # --------old unbounded DP verison-------------------------
    # from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp, get_privacy_spent
    # def compute_epsilon_given_sigma(sigma):
    #     rdp = compute_rdp(q=batch_size / train_x_size0,
    #                       noise_multiplier=sigma,
    #                       steps=epochs * steps_per_epoch,
    #                       orders=orders)

    #     epsilon = get_privacy_spent(orders, rdp, target_delta=delta)[0]
    #     return epsilon
    #-----------------------------------------------------------
    
    # New bounded DP accounting
    def compute_epsilon_given_sigma(sigma):
        rdp = rdp_acc._compute_rdp_sample_wor_gaussian(q=batch_size / train_x_size0,
                        noise_multiplier=sigma,
                        orders=orders)
        steps = epochs * steps_per_epoch
        rdp = rdp * steps
        epsilon = rdp_acc.compute_epsilon(orders, rdp, delta=delta)[0]
        return epsilon
    
    time_init = time.time()
    train_x_size0 = params["train_x_size0"]
    batch_size = params["batch_size"]
    epochs = params["epochs"]
    steps_per_epoch = train_x_size0 // batch_size

    orders = [1 + x / 100.0 for x in range(1, 1000, 5)] + list(range(10, 1200))

    sigma0 = 0.1
    sigma1 = 1000

    epsilon0 = compute_epsilon_given_sigma(sigma0)
    while epsilon0 < target_epsilon:
        sigma0 = sigma0 / 2
        epsilon0 = compute_epsilon_given_sigma(sigma0)
        v_print("Decreasing sigma0: {:.3f}, current epsilon0 = {:.3f}, target = {:.3f} ({:.0f} secs)".format(sigma0, epsilon0, target_epsilon, time.time() - time_init))

    epsilon1 = compute_epsilon_given_sigma(sigma1)
    while epsilon1 > target_epsilon:
        sigma1 = sigma1 * 2
        epsilon1 = compute_epsilon_given_sigma(sigma1)
        v_print("Increasing sigma1: {:.3f}, current epsilon1 = {:.3f}, target = {:.3f} ({:.0f} secs)".format(sigma1, epsilon1, target_epsilon, time.time() - time_init))

    sigma_mid = (sigma0 + sigma1) / 2
    epsilon_mid = compute_epsilon_given_sigma(sigma_mid)
    v_print("Starting iterations with target epsilon = {:.3f}, sigma0 = {:.3f}, sigma1 = {:.3f}".format(target_epsilon, sigma0, sigma1))

    n_iters = 0
    while np.abs(epsilon_mid - target_epsilon) > tolerance and n_iters < 20:
        #If epsilon goes to zero choose the smallest sigma and break
        if epsilon_mid == 0  and sigma1 - sigma0 <= 100:
            break
        n_iters += 1
        if epsilon_mid > target_epsilon:
            sigma0 = sigma_mid
        else:
            sigma1 = sigma_mid
        sigma_mid = (sigma0 + sigma1) / 2
        epsilon_mid = compute_epsilon_given_sigma(sigma_mid)
        v_print("Iter {:d}) sigma = {:.3f}, epsilon = {:.4f}, target epsilon = {:.4f}, error = {:.4f} ({:.0f} secs)".format(n_iters, sigma_mid, epsilon_mid, target_epsilon,
                                                                                                                            np.abs(epsilon_mid - target_epsilon), time.time() - time_init))
    v_print("Done! For epsilon = {:.3f}, sigma = {:.3f} ({:.0f} secs)".format(target_epsilon, sigma_mid, time.time() - time_init))
    return sigma_mid
    
def get_params(key):
    tags = key.split('_')
    params = { 'train_x_size0' : int(tags[0]),
              'batch_size' : int(tags[1]),
              'epochs' : int(tags[2]),
              'delta' : float(tags[3])
    }
    return params

def main():
    # In Jayaraman, for train_x_size = 5000: noise_multiplier = {0.01:525, 0.05:150, 0.1:70, 0.5:13.8, 1:7, 5:1.669, 10:1.056, 50:0.551, 100:0.445, 500:0.275, 1000:0.219}
    # For 384: {0.05: 433.650390625, 0.1: 216.8751953125, 0.5: 43.85783081054688, 1: 22.215402984619143, 5: 4.885535955429076, 10: 2.7094687938690187, 50: 0.909111738204956, 100: 0.6344810009002686, 500: 0.30644984245300294, 1000: 0.22634921073913572}
    ## INPUT PARAMETERS
    # 
    
    #------------Regenerate current file using new accounting---------------
    with open('unbounded_noise_multipliers.p', 'rb') as f:
        old_noise_multipliers = pickle.load(f)

    new_noise_multipliers = {}
    for config in old_noise_multipliers:
        print(config)
        params = get_params(config)
        temp = {}
        for eps in old_noise_multipliers[config]:
            sigma = compute_sigma_given_epsilon_bisection(eps, params['delta'], 1e-4, params, verbose=True)
            temp[eps] = sigma
        new_noise_multipliers[config] = temp

    with open(NOISE_MULT_FILE, 'wb') as f:
        pickle.dump(new_noise_multipliers, f)
    #----------------------------------------------------------------------


if __name__ == "__main__":
    main()
