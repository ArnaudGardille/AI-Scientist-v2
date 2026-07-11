# Target-policy replay research

Improve off-policy cooperative multi-agent learning when teammate policies change while
the conditional environment dynamics remain stationary given the joint action.

The operator tier rewards accurate estimation of the current perceived Bellman backup
under behavior/target policy mismatch. Promising directions include exact importance
sampling, clipped or ESS-controlled ratios, and action-stratified conditional means.

The learning tier tests whether an estimator translates into useful replay sampling on
cooperative STORM. Simultaneous-Attack is a strict support-preservation gate: rare optimal
joint actions must not be filtered out. The candidate must preserve the existing public
functions and may only change `candidate.py`.
