import numpy as np
import pandas as pd


def read_stat_csv(stat_csv_file, feature_columns=None):
    file_lines = open(stat_csv_file).readlines()
    numeric_column = ["secs_elapsed", "disk_usage", "cpu_utils"]
    # string_column = ["db_path"]
    default_columns = ["secs_elapsed", "db_path", "disk_usage", "cpu_utils"]

    result = []
    for file_line in file_lines:
        records = file_line.replace("\n", "").split(",")
        df_records = records[0:len(default_columns)]
        result.append(df_records)
    result_df = pd.DataFrame(result, columns=default_columns)
    result_df[numeric_column] = result_df[numeric_column].astype(float)
    if feature_columns:
        return result_df[feature_columns]
    else:
        return result_df


def read_report_csv_with_change_points(report_file):
    file_lines = open(report_file).readlines()
    result = []
    for file_line in file_lines[1:]:
        records = file_line.replace("\n", "").split(",")
        if (len(records) > 2):
            normed = int(records[3]) / (64 * 1024 * 1024)
            records = records[0:2]
            records.append(normed)
        else:
            records.append(1)
        result.append(records)
    result = np.array(result).astype(float)
    return pd.DataFrame(result, columns=["secs_elapsed", "interval_qps", "change_points"])


def action_list_feature_vectorize(log_and_qps, time_slice):
    ms_to_second = 1000000
    # max_file_level = 7
    feature_columns = ["flushes", "l0compactions",
                       "other_compactions", "read", "write"]
    # file_counter_list = ["level"+str(x) for x in range(max_file_level)]
    file_counter_list = []
    feature_columns.extend(file_counter_list)

    switch_ratio = ms_to_second / time_slice

    real_time_speed = log_and_qps.qps_df

    elasped_time = int(real_time_speed.tail(1)["secs_elapsed"] * switch_ratio)

    bucket = np.zeros([elasped_time, len(feature_columns)], dtype=float)
    for index, flush_job in log_and_qps.flush_df.iterrows():
        # bytes/ms , equals to MB/sec
        flush_speed = round(
            flush_job["flush_size"] / (flush_job["end_time"] - flush_job["start_time"]), 2)
        start_index = int(flush_job["start_time"] / time_slice)
        end_index = int(flush_job["end_time"] / time_slice) + 1
        # the tail part is not accurant
        if start_index >= len(bucket) - 10 or end_index >= len(bucket) - 5:
            break
        for element in bucket[start_index:end_index]:
            element[0] += 1
            element[4] += flush_speed

    for index, compaction_job in log_and_qps.compaction_df.iterrows():
        compaction_read_speed = round(compaction_job["input_data_size"] / (
            compaction_job["compaction_time_micros"]), 2)  # bytes/ms , equals to MB/sec
        compaction_write_speed = round(compaction_job["total_output_size"] / (
            compaction_job["compaction_time_micros"]), 2)  # bytes/ms , equals to MB/sec
        start_index = int(compaction_job["start_time"] / time_slice)
        end_index = int(compaction_job["end_time"] / time_slice) + 1
        lsm_state = compaction_job["lsm_state"]

        # the tail part is not accurant
        if start_index >= len(bucket) - 10 or end_index >= len(bucket) - 5:
            break
        for element in bucket[start_index:end_index]:
            element[0] += 0
            if compaction_job["compaction_reason"] == "LevelL0FilesNum":
                element[1] += 1
            else:
                element[2] += 1
            element[3] += compaction_read_speed
            element[4] += compaction_write_speed
            for level in range(len(file_counter_list)):
                # print(lsm_state[level])
                element[5 + level] += lsm_state[level]
                # print(level)
    # compute the mean of the lsm state

    return pd.DataFrame(bucket, columns=feature_columns)


def vectorize_by_compaction_output_level(log_and_qps, time_slice=1000000):
    ms_to_second = 1000000
    max_file_level = 4
    feature_columns = ["flushes", "l0compactions",
                       "other_compactions", "read", "write"]
    file_counter_list = ["level" + str(x) for x in range(max_file_level)]
    # file_counter_list = []
    feature_columns.extend(file_counter_list)

    switch_ratio = ms_to_second / time_slice

    real_time_speed = log_and_qps.qps_df

    elasped_time = int(real_time_speed.tail(1)["secs_elapsed"] * switch_ratio)

    bucket = np.zeros([elasped_time, len(feature_columns)], dtype=float)
    for index, flush_job in log_and_qps.flush_df.iterrows():
        # bytes/ms , equals to MB/sec
        flush_speed = round(
            flush_job["flush_size"] / (flush_job["end_time"] - flush_job["start_time"]), 2)
        start_index = int(flush_job["start_time"] / time_slice)
        end_index = int(flush_job["end_time"] / time_slice) + 1
        # the tail part is not accurant
        if start_index >= len(bucket) - 10 or end_index >= len(bucket) - 5:
            break
        for element in bucket[start_index:end_index]:
            element[0] += 1
            element[4] += flush_speed

    for index, compaction_job in log_and_qps.compaction_df.iterrows():
        compaction_read_speed = round(compaction_job["input_data_size"] / (
            compaction_job["compaction_time_micros"]), 2)  # bytes/ms , equals to MB/sec
        compaction_write_speed = round(compaction_job["total_output_size"] / (
            compaction_job["compaction_time_micros"]), 2)  # bytes/ms , equals to MB/sec
        start_index = int(compaction_job["start_time"] / time_slice)
        end_index = int(compaction_job["end_time"] / time_slice) + 1

        # the tail part is not accurant
        if start_index >= len(bucket) - 10 or end_index >= len(bucket) - 5:
            break
        for element in bucket[start_index:end_index]:
            element[0] += 0
            if compaction_job["compaction_reason"] == "LevelL0FilesNum":
                element[1] += 1
            else:
                element[2] += 1
            element[3] += compaction_read_speed
            element[4] += compaction_write_speed
            for level in range(len(file_counter_list)):
                # print(compaction_job["output_level"])
                if compaction_job["output_level"] == level:
                    element[5 + level] += 1
                # print(level)
    # compute the mean of the lsm state
    return pd.DataFrame(bucket, columns=feature_columns)


def vectorize_by_disk_op_distribution(log_and_qps, time_slice=1000000):
    ms_to_second = 1000000
    feature_columns = ["flushes", "l0compactions",
                       "other_compactions", "read", "write", "write_flush", "write_compaction"]
    switch_ratio = ms_to_second / time_slice

    real_time_speed = log_and_qps.qps_df

    elasped_time = int(real_time_speed.tail(1)["secs_elapsed"] * switch_ratio)

    bucket = np.zeros([elasped_time, len(feature_columns)], dtype=float)
    for index, flush_job in log_and_qps.flush_df.iterrows():
        # bytes/ms , equals to MB/sec
        flush_speed = round(
            flush_job["flush_size"] / (flush_job["end_time"] - flush_job["start_time"]), 2)
        start_index = int(flush_job["start_time"] / time_slice)
        end_index = int(flush_job["end_time"] / time_slice) + 1
        # the tail part is not accurant
        if start_index >= len(bucket) - 10 or end_index >= len(bucket) - 5:
            break
        for element in bucket[start_index:end_index]:
            element[0] += 1
            element[4] += flush_speed
            element[5] += flush_speed

    for index, compaction_job in log_and_qps.compaction_df.iterrows():
        compaction_read_speed = round(compaction_job["input_data_size"] / (
            compaction_job["compaction_time_micros"]), 2)  # bytes/ms , equals to MB/sec
        compaction_write_speed = round(compaction_job["total_output_size"] / (
            compaction_job["compaction_time_micros"]), 2)  # bytes/ms , equals to MB/sec
        start_index = int(compaction_job["start_time"] / time_slice)
        end_index = int(compaction_job["end_time"] / time_slice) + 1

        # the tail part is not accurant
        if start_index >= len(bucket) - 10 or end_index >= len(bucket) - 5:
            break
        for element in bucket[start_index:end_index]:
            element[0] += 0
            if compaction_job["compaction_reason"] == "LevelL0FilesNum":
                element[1] += 1
            else:
                element[2] += 1
            element[3] += compaction_read_speed
            element[4] += compaction_write_speed
            element[6] += compaction_write_speed
    # compute the mean of the lsm state
    return pd.DataFrame(bucket, columns=feature_columns)


def generate_lsm_shape(log_and_qps, level=7, time_slice=1000000):
    # result = None
    columns = ["job_id", "op_type", "time_micro"]  # job_id, op, []
    for i in range(level):
        columns.append("level" + str(i))
    result = []
    for index, compaction_job in log_and_qps.compaction_df.iterrows():
        # print(compaction_job)
        moment_lsm_shape = [compaction_job["job"], "compaction", float(compaction_job["end_time"]) / time_slice]
        for count in compaction_job["lsm_state"][0:level]:
            moment_lsm_shape.append(count)
        result.append(moment_lsm_shape)
    # for index, flush_job in log_and_qps.flush_df.iterrows():
    #     # print(compaction_job)
    #     moment_lsm_shape = [flush_job["job"], "flush", flush_job["end_time"]]
    #     for count in flush_job["lsm_state"][0:level]:
    #         moment_lsm_shape.append(count)
    #     result.append(moment_lsm_shape)
    result = pd.DataFrame(result, columns=columns)
    result = result.sort_values(by=['time_micro'])
    result = result.reset_index(drop=True)
    return result


def combine_vector_with_qps(bucket_df, qps_df):
    # since qps_df starts from sec 1, add the first line, [0,0,0]
    id_df = pd.DataFrame(list(range(bucket_df.shape[0])), columns=["secs_elapsed"])
    id_df = id_df.merge(qps_df, how="left", on="secs_elapsed")
    id_df['interval_qps'] = id_df['interval_qps'].fillna(0)
    id_df['change_points'] = id_df['change_points'].fillna(1)
    id_df = id_df[["interval_qps", "change_points"]]
    result_bf = pd.concat([bucket_df, id_df], axis=1)
    return result_bf
