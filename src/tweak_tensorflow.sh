#!/usr/bin/bash

tp_folder=$(python -c "import tensorflow_privacy as _; print('path', _.__path__[0])" | grep -oP 'path (.+)' | cut -d ' ' -f 2 -)
rdp_accountant=$tp_folder"/privacy/analysis/rdp_accountant.py"

echo "Modifying tensorflow_privacy at $rdp_accountant ..."


old_regex="def _compute_log_a_int.+?return float\(log_a\)"

# Basically, the old version of this function is extremely slow to compute. We are replacing it with a more
# efficient solution. This is technically only necessary if you are planning to run the code located in
# compute_noise.py as it is the bisection search that causes the slowness.
new_code="def _compute_log_a_int(q, sigma, alpha):
  assert isinstance(alpha, six.integer_types)
  log_a = -np.inf
  log_binom = 0
  for i in range(alpha + 1):
    log_coef_i = (log_binom + i * math.log(q) + (alpha - i) * math.log(1 - q))
    if i < alpha: log_binom += math.log(alpha - i) - math.log(i + 1)
    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)
  return float(log_a)"

perl -0777 -i.original -pe "s|$old_regex|$new_code|igs" $rdp_accountant
