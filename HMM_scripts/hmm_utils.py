import matplotlib.pyplot as plt
import numpy as np

from tensorflow_probability import  distributions as tfd
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

def plot_by_num_of_states(bucket_df, list_of_moment_map, num_of_states):
    moment_map = list_of_moment_map[num_of_states]

    for moment_list in moment_map:
        if len(moment_map[moment_list]) > 0:
            # bucket_df[moment_map[moment_list]]
            selected_df = bucket_df.loc[moment_map[moment_list]].reindex()
            selected_df.plot(subplots=True)
            # bucket_df["qps"][moment_map[moment_list]].plot()


def get_list_of_moment_map(fitting_area):
    def build_latent_state(num_states, max_num_states, daily_change_prob=0.05):

        # Give probability exp(-100) ~= 0 to states outside of the current model.
        initial_state_logits = -100. * np.ones([max_num_states], dtype=np.float32)
        initial_state_logits[:num_states] = 0.
        initial_state_logits[0] = 1.
        # Build a transition matrix that transitions only within the current
        # `num_states` states.
        transition_probs = np.eye(max_num_states, dtype=np.float32)
        if num_states > 1:
            transition_probs[:num_states, :num_states] = (
                    daily_change_prob / (num_states - 1))
            np.fill_diagonal(transition_probs[:num_states, :num_states],
                             1 - daily_change_prob)
        return initial_state_logits, transition_probs

    max_num_states = 10
    batch_initial_state_logits = []
    batch_transition_probs = []
    for num_states in range(1, max_num_states + 1):
        initial_state_logits, transition_probs = build_latent_state(
            num_states=num_states,
            max_num_states=max_num_states)
        batch_initial_state_logits.append(initial_state_logits)
        batch_transition_probs.append(transition_probs)
    batch_initial_state_logits = np.array(batch_initial_state_logits)
    batch_transition_probs = np.array(batch_transition_probs)

    trainable_log_rates = tf.Variable(
        (np.log(np.mean(fitting_area)) *
         np.ones([batch_initial_state_logits.shape[0], max_num_states]) +
         tf.random.normal([1, max_num_states])),
        name='log_rates')

    hmm = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(
            logits=batch_initial_state_logits),
        transition_distribution=tfd.Categorical(probs=batch_transition_probs),
        observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
        num_steps=len(fitting_area))
    rate_prior = tfd.LogNormal(5, 5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    def log_prob():
        prior_lps = rate_prior.log_prob(tf.math.exp(trainable_log_rates))
        prior_lp = tf.stack(
            [tf.reduce_sum(prior_lps[i, :i + 1]) for i in range(max_num_states)])
        return prior_lp + hmm.log_prob(fitting_area)

    @tf.function(autograph=False)
    def train_op():
        with tf.GradientTape() as tape:
            neg_log_prob = -log_prob()
        grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
        optimizer.apply_gradients([(grads, trainable_log_rates)])
        return neg_log_prob, tf.math.exp(trainable_log_rates)

    for step in range(201):
        loss, rates = [t.numpy() for t in train_op()]
        if step % 20 == 0:
            print("step {}: loss {}".format(step, loss))

    posterior_probs = hmm.posterior_marginals(
        fitting_area).probs_parameter().numpy()
    most_probable_states = np.argmax(posterior_probs, axis=-1)

    fig = plt.figure(figsize=(14, 12))
    for i, learned_model_rates in enumerate(rates):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.plot(learned_model_rates[most_probable_states[i]], c='green', lw=3, label='inferred rate')
        ax.plot(fitting_area, c='black', alpha=0.3, label='observed counts')
        ax.set_ylabel("latent rate")
        ax.set_xlabel("time")
        ax.set_title("{}-state model".format(i + 1))
        ax.legend(loc=4)
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(14, 12))
    list_of_moment_map = []
    for number_of_states in range(max_num_states):
        moment_map = {}
        ax = fig.add_subplot(4, 3, number_of_states + 1)
        for state_no in range(max_num_states):
            moment_map[state_no] = []

        index = 0
        for state in most_probable_states[number_of_states]:
            moment_map[state].append(index)
            index += 1
        # moment_map = {k:v for k,v in moment_map.items() if len(v) > 0}
        frequency_count = [len(moment_map[x]) / index for x in moment_map]
        bar1 = ax.bar(range(len(moment_map)), frequency_count)
        # autolabel(bar1,most_probable_states[number_of_states])
        ax.set_ylim(0, 1.1)
        ax.set_xlabel("state id")
        ax.set_title("{}-state model".format(i + 1))
        list_of_moment_map.append(moment_map)
    plt.tight_layout()
    plt.savefig("rate_frequency.png")
    plt.clf()

    return list_of_moment_map


def latent_state_number_changing_curve(fitting_area, output_dir_prefix, log_dir_prefix, log_dir, fig_name=""):
    max_num_states = 10

    def build_latent_state(num_states, max_num_states, daily_change_prob=0.05):

        # Give probability exp(-100) ~= 0 to states outside of the current model.
        initial_state_logits = -100. * np.ones([max_num_states], dtype=np.float32)
        initial_state_logits[:num_states] = 0.
        initial_state_logits[0] = 1.
        # Build a transition matrix that transitions only within the current
        # `num_states` states.
        transition_probs = np.eye(max_num_states, dtype=np.float32)
        if num_states > 1:
            transition_probs[:num_states, :num_states] = (
                    daily_change_prob / (num_states - 1))
            np.fill_diagonal(transition_probs[:num_states, :num_states],
                             1 - daily_change_prob)
        return initial_state_logits, transition_probs

    # For each candidate model, build the initial state prior and transition matrix.
    batch_initial_state_logits = []
    batch_transition_probs = []
    for num_states in range(1, max_num_states + 1):
        initial_state_logits, transition_probs = build_latent_state(
            num_states=num_states,
            max_num_states=max_num_states)
        batch_initial_state_logits.append(initial_state_logits)
        batch_transition_probs.append(transition_probs)
    batch_initial_state_logits = np.array(batch_initial_state_logits)
    batch_transition_probs = np.array(batch_transition_probs)

    trainable_log_rates = tf.Variable(
        (np.log(np.mean(fitting_area)) *
         np.ones([batch_initial_state_logits.shape[0], max_num_states]) +
         tf.random.normal([1, max_num_states])),
        name='log_rates')

    hmm = tfd.HiddenMarkovModel(
        initial_distribution=tfd.Categorical(
            logits=batch_initial_state_logits),
        transition_distribution=tfd.Categorical(probs=batch_transition_probs),
        observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
        num_steps=len(fitting_area))
    rate_prior = tfd.LogNormal(5, 5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    def log_prob():
        prior_lps = rate_prior.log_prob(tf.math.exp(trainable_log_rates))
        prior_lp = tf.stack(
            [tf.reduce_sum(prior_lps[i, :i + 1]) for i in range(max_num_states)])
        return prior_lp + hmm.log_prob(fitting_area)

    @tf.function(autograph=False)
    def train_op():
        with tf.GradientTape() as tape:
            neg_log_prob = -log_prob()
        grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
        optimizer.apply_gradients([(grads, trainable_log_rates)])
        return neg_log_prob, tf.math.exp(trainable_log_rates)

    for step in range(201):
        loss, rates = [t.numpy() for t in train_op()]
        if step % 20 == 0:
            print("step {}: loss {}".format(step, loss))

    num_states = np.arange(1, max_num_states + 1)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(num_states, -loss, "b-", label="likelihood")
    plt.ylabel("marginal likelihood $\\tilde{p}(x)$")
    plt.xlabel("number of latent states")
    plt.legend()
    plt.twinx()
    plt.plot(num_states, np.gradient(-loss), "g--", label="gradient")
    plt.ylabel("Gradient of the likelihood")
    plt.title("Model selection on latent states")
    plt.legend()
    output_path = output_dir_prefix + log_dir.replace(log_dir_prefix, "").replace("/", "_")
    mkdir_p(output_path)
    plt.savefig("{}/{}_likelihood_curve.pdf".format(output_path, fig_name), bbox_inches="tight")
    plt.savefig("{}/{}_likelihood_curve.png".format(output_path, fig_name), bbox_inches="tight")
    plt.clf()

    posterior_probs = hmm.posterior_marginals(
        fitting_area).probs_parameter().numpy()
    most_probable_states = np.argmax(posterior_probs, axis=-1)

    fig = plt.figure(figsize=(14, 12))
    for i, learned_model_rates in enumerate(rates):
        ax = fig.add_subplot(4, 3, i + 1)
        ax.plot(learned_model_rates[most_probable_states[i]], c='green', lw=3, label='inferred rate')
        ax.plot(fitting_area, c='black', alpha=0.3, label='observed counts')
        ax.set_ylabel("latent rate")
        ax.set_xlabel("time")
        ax.set_title("{}-state model".format(i + 1))
        ax.legend(loc=4)
    plt.tight_layout()
    plt.savefig("{}/{}_model_fitting_test.pdf".format(output_path, fig_name), bbox_inches="tight")
    plt.savefig("{}/{}_model_fitting_test.png".format(output_path, fig_name), bbox_inches="tight")
    plt.clf()
    pass
