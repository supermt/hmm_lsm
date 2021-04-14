import numpy as np
import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from matplotlib import pylab as plt
import scipy.stats
import tensorflow as tfv

if __name__ == '__main__':
    print(tfv.__version__)
    print(tfp.__version__)
    num_states = 4

    true_rates = [40, 3, 20, 50]
    true_durations = [10, 20, 5, 35]

    observed_counts = np.concatenate([
        scipy.stats.poisson(rate).rvs(num_steps)
        for (rate, num_steps) in zip(true_rates, true_durations)
    ]).astype(np.float32)

    plt.plot(observed_counts)
    plt.show()

    initial_state_logits = np.zeros([num_states], dtype=np.float32)  # uniform distribution

    daily_change_prob = 0.05
    transition_probs = daily_change_prob / (num_states - 1) * np.ones(
        [num_states, num_states], dtype=np.float32)
    np.fill_diagonal(transition_probs,
                     1 - daily_change_prob)

    print("Initial state logits:\n{}".format(initial_state_logits))
    print("Transition matrix:\n{}".format(transition_probs))
    trainable_log_rates = tf.Variable(
        np.log(np.mean(observed_counts)) + tf.random.normal([num_states]),
        name='log_rates')

    hmm = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(
            logits=initial_state_logits),
        transition_distribution=tfd.Categorical(probs=transition_probs),
        observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
        num_steps=len(observed_counts))
