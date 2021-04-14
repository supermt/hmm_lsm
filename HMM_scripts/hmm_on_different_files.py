import numpy as np
import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()

from tensorflow_probability import distributions as tfd

from matplotlib import pylab as plt

from compaction_distribution import get_log_dirs
from compaction_distribution import get_log_and_std_files
from feature_selection import vectorize_by_compaction_output_level
from compaction_distribution import load_log_and_qps
from traversal import mkdir_p

from hmm_utils import latent_state_number_changing_curve


def two_period_fitting(log_dir, log_dir_prefix="log_files/", output_dir_prefix="image/"):
    stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)

    data_set = load_log_and_qps(LOG_file, report_csv)
    bucket_df = vectorize_by_compaction_output_level(data_set)
    bucket_df["qps"] = data_set.qps_df["interval_qps"]

    observed_counts = bucket_df["qps"].fillna(0).tolist()
    observed_counts = np.array(observed_counts).astype(np.float32)

    latent_state_number_changing_curve(observed_counts[201:600], output_dir_prefix, log_dir_prefix, log_dir,
                                       "201_to_600_period")
    latent_state_number_changing_curve(observed_counts[601:1000], output_dir_prefix, log_dir_prefix, log_dir,
                                       "601_to_1000_period")


def choose_HMM_state_numbers(log_dir, log_dir_prefix="log_files/", output_dir_prefix="image/"):
    stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)

    data_set = load_log_and_qps(LOG_file, report_csv)
    bucket_df = vectorize_by_compaction_output_level(data_set)
    bucket_df["qps"] = data_set.qps_df["interval_qps"]

    observed_counts = bucket_df["qps"].fillna(0).tolist()
    observed_counts = np.array(observed_counts).astype(np.float32)

    latent_state_number_changing_curve(observed_counts, output_dir_prefix, log_dir_prefix)


def HMM_on_one_file(log_dir):
    stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)

    data_set = load_log_and_qps(LOG_file, report_csv)
    bucket_df = vectorize_by_compaction_output_level(data_set)
    bucket_df["qps"] = data_set.qps_df["interval_qps"]

    _ = bucket_df.plot(subplots=True)
    num_states = 5  # memtable filling, flush only, L0 compaction (CPU busy), crowded compaction (disk busy)

    initial_state_logits = np.zeros([num_states], dtype=np.float32)  # uniform distribution

    initial_state_logits[0] = 1.0  # the possiblity of transferring into the Flushing limitation
    initial_state_logits

    initial_distribution = tfd.Categorical(probs=initial_state_logits)

    daily_change_prob = 0.05
    transition_probs = daily_change_prob / (num_states - 1) * np.ones(
        [num_states, num_states], dtype=np.float32)
    np.fill_diagonal(transition_probs,
                     1 - daily_change_prob)

    observed_counts = bucket_df["qps"].fillna(0).tolist()
    observed_counts = np.array(observed_counts).astype(np.float32)

    transition_distribution = tfd.Categorical(probs=transition_probs)
    trainable_log_rates = tf.Variable(
        np.log(np.mean(observed_counts)) + tf.random.normal([num_states]),
        name='log_rates')

    hmm = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
        num_steps=len(observed_counts))

    rate_prior = tfd.LogNormal(5, 5)

    #
    def log_prob():
        return (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) +
                hmm.log_prob(observed_counts))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function(autograph=False)
    def train_op():
        with tf.GradientTape() as tape:
            neg_log_prob = -log_prob()
        grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
        optimizer.apply_gradients([(grads, trainable_log_rates)])
        return neg_log_prob, tf.math.exp(trainable_log_rates)

    #
    for step in range(201):
        loss, rates = [t.numpy() for t in train_op()]
        if step % 20 == 0:
            print("step {}: log prob {} rates {}".format(step, -loss, rates))

    posterior_dists = hmm.posterior_marginals(observed_counts)
    posterior_probs = posterior_dists.probs_parameter().numpy()
    most_probable_states = np.argmax(posterior_probs, axis=1)
    most_probable_rates = rates[most_probable_states]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(most_probable_rates, c='green', lw=3, label='inferred rate')
    ax.plot(observed_counts, c='black', alpha=0.3, label='observed counts')
    ax.set_ylabel("latent rate")
    ax.set_xlabel("time")
    ax.set_title("Inferred latent rate over time")
    ax.legend(loc=4)
    output_path = "image/" + log_dir.replace("log_files/", "").replace("/", "_")
    mkdir_p(output_path)
    plt.savefig("{}/state_guessing.pdf".format(output_path), bbox_inches="tight")


def HMM_train_and_predict(log_dir, log_dir_prefix, output_prefx="predction_test/"):
    stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)

    data_set = load_log_and_qps(LOG_file, report_csv)
    bucket_df = vectorize_by_compaction_output_level(data_set)
    bucket_df["qps"] = data_set.qps_df["interval_qps"]

    # 200:800 training set
    # 801:eof prediction set

    output_path = output_prefx + log_dir.replace(log_dir_prefix, "").replace("/", "_")
    mkdir_p(output_path)
    _ = bucket_df.plot(subplots=True)
    plt.savefig("{}/compaction_stats.pdf".format(output_path), bbox_inches="tight")
    plt.clf()
    num_states = 5  # memtable filling, flush only, L0 compaction (CPU busy), crowded compaction (disk busy)

    initial_state_logits = np.zeros([num_states], dtype=np.float32)  # uniform distribution

    initial_state_logits[0] = 1.0  # the possiblity of transferring into the Flushing limitation
    initial_state_logits

    initial_distribution = tfd.Categorical(probs=initial_state_logits)

    daily_change_prob = 0.05
    transition_probs = daily_change_prob / (num_states - 1) * np.ones(
        [num_states, num_states], dtype=np.float32)
    np.fill_diagonal(transition_probs,
                     1 - daily_change_prob)

    data_array = bucket_df["qps"].fillna(0).tolist()
    data_array = np.array(data_array).astype(np.float32)

    observed_counts = data_array[201:600]
    prediction_area = data_array[601:1000]

    transition_distribution = tfd.Categorical(probs=transition_probs)
    trainable_log_rates = tf.Variable(
        np.log(np.mean(observed_counts)) + tf.random.normal([num_states]),
        name='log_rates')

    hmm = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=tfd.Poisson(log_rate=trainable_log_rates),
        num_steps=len(observed_counts))

    rate_prior = tfd.LogNormal(5, 5)

    #
    def log_prob():
        return (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) +
                hmm.log_prob(observed_counts))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function(autograph=False)
    def train_op():
        with tf.GradientTape() as tape:
            neg_log_prob = -log_prob()
        grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
        optimizer.apply_gradients([(grads, trainable_log_rates)])
        return neg_log_prob, tf.math.exp(trainable_log_rates)

    #
    for step in range(201):
        loss, rates = [t.numpy() for t in train_op()]
        if step % 20 == 0:
            print("step {}: log prob {} rates {}".format(step, -loss, rates))

    posterior_dists_fitting = hmm.posterior_marginals(observed_counts)
    most_probable_rates_fitting = rates[np.argmax(posterior_dists_fitting.probs_parameter().numpy(), axis=1)]

    posterior_dists_prediction = hmm.posterior_marginals(prediction_area)
    most_probable_rates_prediction = rates[np.argmax(posterior_dists_prediction.probs_parameter().numpy(), axis=1)]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(range(len(observed_counts)), most_probable_rates_fitting, c='blue', linestyle="--",
            lw=3, label='inferred rate (fitting period)')
    ax.plot(range(len(observed_counts)), observed_counts.tolist(), c='black', alpha=0.3, label='observed counts')

    ax.plot(range(len(observed_counts), len(observed_counts) + len(prediction_area)), most_probable_rates_prediction,
            c='green', linestyle="-",
            lw=3, label='inferred rate (prediction period)')
    ax.plot(range(len(observed_counts), len(observed_counts) + len(prediction_area)), prediction_area.tolist(), c='red',
            alpha=0.3,
            label='prediction ground truth')
    ax.set_ylabel("latent rate")
    ax.set_xlabel("time")
    ax.set_title("Inferred latent rate over time")
    ax.legend(loc=4)
    plt.savefig("{}/state_guessing.pdf".format(output_path), bbox_inches="tight")
    print("fig saved at {}".format(output_path))
    plt.clf()


if __name__ == '__main__':
    log_prefix_dir = "log_files"
    dirs = get_log_dirs(log_prefix_dir)

    for log_dir in dirs:
        # choose_HMM_state_numbers(log_dir, log_prefix_dir, "middle_change_log_statistics/")
        # HMM_train_and_predict(log_dir, log_prefix_dir, "no_middle_change_log_predict/")
        two_period_fitting(log_dir, log_prefix_dir, "no_middle_change_log_predict/")

    log_prefix_dir = "../middle_changing_log"
    dirs = get_log_dirs(log_prefix_dir)

    for log_dir in dirs:
        # choose_HMM_state_numbers(log_dir, log_prefix_dir, "middle_change_log_statistics/")
        # HMM_train_and_predict(log_dir, log_prefix_dir, "no_middle_change_log_predict/")
        two_period_fitting(log_dir, log_prefix_dir, "middle_changed_log_predict/")
