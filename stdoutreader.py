import re

from traversal import get_log_dirs, get_log_and_std_files

BENCHMARK_RESULT_REGEX = r"[a-zA-z]+\s+:\s+[0-9]+\.[0-9]+\smicros\/op\s[0-9]+\sops\/sec;\s+.+"

source_file = "LOG_DIR/trade_off_analysis/StorageMaterial.NVMeSSD/1CPU/64MB/stdout.txt_6241"

COMPACTION_STAT_METRICS = ["Flush(GB)", "Cumulative compaction", "Interval compaction", "Stalls(count)"]

DB_STAT_METRICS = ["Cumulative writes"]

COMPACTION_COLUMN_NAME_MAP = {"GB write": "write size", "MB/s write": "write throughput",
                              "GB read": "read write", "MB/s read": "read throughput",
                              "seconds": "time"}

READING_LINE = "** Level"

STATISTICS_MEASURES = ["cache.miss", "cache.hit", "cache.add"]


def get_benchmark_entry(line):
    benchmark = line.split(":")[0].replace(" ", "")
    metrics_reg = r"[0-9]+(\.)*[0-9]+\s[a-zA-Z]+\/[a-zA-Z]+"
    matches = re.finditer(metrics_reg, line, re.MULTILINE)
    metrics = []
    end_num = 0
    for matchNum, match in enumerate(matches, start=1):
        metrics.append(match.group())
        end_num = match.end()

    if end_num == len(line):
        notes = ""
    else:
        notes = line[end_num:]
    metrics.append(notes)
    return {benchmark: metrics}


def get_db_stat_entry(line):
    regex = r"([0-9]+\.)*[0-9]+[a-zA-Z]*\swrites"
    matches = re.search(regex, line, re.M | re.I)
    total_write_op = matches.group()[:-7]
    ingest_metrics = line.split("ingest:")[1].split(",")
    temp_list = ingest_metrics[0].split(" ")
    total_wirte_size = float(temp_list[1])
    total_wirte_unit = temp_list[2]
    if total_wirte_unit == "GB":
        total_wirte_size *= 1024
    elif total_wirte_unit == "MB":
        total_wirte_size *= 1
    return {"total_write_op": total_write_op, "total_write_size": total_wirte_size}


def get_compaction_stat_entry(line, entry_name):
    line = line.replace(entry_name + ":", "")
    if entry_name == "Flush(GB)":
        return {entry_name: float(line.split(" ")[2][:-1])}
    elif entry_name == "Stalls(count)":
        result_map = {}
        different_stalls = line.split(",")[:-1]
        for stall in different_stalls:
            stall_num_matches = re.search(r"[0-9]+", stall)
            stall_num = int(stall_num_matches.group(0))
            column_name = stall[stall_num_matches.end(0) + 1:]
            result_map[column_name] = stall_num
        return result_map
    else:
        # Cumulative compaction and Interval compaction
        result_map = {}
        line = line.replace(entry_name + ":", "")
        compaction_metrics = line.split(",")
        for metrics in compaction_metrics:
            match = re.search(r"([0-9]+\.)*[0-9]+", metrics)
            numeric_data = float(match.group(0))
            column_name = metrics[match.end(0) + 1:]
            column_name = entry_name + " " + COMPACTION_COLUMN_NAME_MAP[column_name]
            result_map[column_name] = numeric_data
        return result_map


def get_read_latency_entry(line):
    result_map = {}

    regex = r"[a-zA-Z]+:\s([0-9]+\.)*[0-9]+"
    matches = re.finditer(regex, line, re.MULTILINE)

    for matchNum, match in enumerate(matches, start=1):
        result_entry = match.group().split(":")
        result_map[result_entry[0]] = result_entry[1]
    return result_map


class StdoutReader:
    def split_the_file(self):
        line_counter = 0

        for line in self.filelines:
            matchObj_benchmark = re.match(BENCHMARK_RESULT_REGEX, line)

            if matchObj_benchmark:
                self.line_map["benchmarks"].append(line_counter)
                self.benchmark_results.update(get_benchmark_entry(line))

            for compaction_metrics in COMPACTION_STAT_METRICS:
                if compaction_metrics in line:
                    self.line_map["compaction_stat"].append(line_counter)
                    temp_map = get_compaction_stat_entry(line, compaction_metrics)
                    self.tradeoff_data.update(temp_map)

            for db_metrics in DB_STAT_METRICS:
                if db_metrics in line:
                    self.line_map["db_stats"].append(line_counter)
                    self.tradeoff_data.update(get_db_stat_entry(line))

            if READING_LINE in line:
                level_key = "level" + line[9]
                self.read_latency_map[level_key].update(get_read_latency_entry(self.filelines[line_counter + 1]))
            for measure in STATISTICS_MEASURES:
                if measure in line:
                    self.tradeoff_data[measure] = int(line.split(":")[1][1:])

            line_counter += 1
        pass

    def __init__(self, input_file):
        self.stdout_file = input_file
        self.filelines = [x.replace("\n", "") for x in open(input_file, "r").readlines()]
        self.benchmark_results = {}
        self.tradeoff_data = {}
        # add basic information
        base_info = self.stdout_file.split("/")
        self.cpu_count = base_info[-3]
        self.batch_size = base_info[-2]
        self.device = base_info[-4].replace("StorageMaterial.","")

        marked_lines = ["benchmarks", "compaction_stat", "file_reading", "db_stats",
                        "statistics"]
        self.line_map = {x: [] for x in marked_lines}

        # extracting read latency
        level_num = 7
        self.read_latency_map = {"level" + str(x): {"Count": 0, "Average": 0, "StdDev": 0} for x in range(level_num)}

        # handle the file
        self.split_the_file()
        return


if __name__ == '__main__':
    tradeoff_data = []
    StdoutReader(source_file)
    log_dir_prefix = "LOG_DIR/trade_off_analysis"
    dirs = get_log_dirs(log_dir_prefix)

    metrics_in_std_files = []
    for log_dir in dirs:
        stdout_file, LOG_file, report_csv = get_log_and_std_files(log_dir)
        metrics_in_std_files.append(StdoutReader(stdout_file))

    result_dict_list = []
    filed_names = []
    for std_record in metrics_in_std_files:
        temp_dict = {}
        temp_dict["threads"] = std_record.cpu_count
        temp_dict["material"] = std_record.device
        temp_dict["batch_size"] = std_record.batch_size
        temp_dict.update(std_record.tradeoff_data)
        for level in std_record.read_latency_map:
            for level_entry in std_record.read_latency_map[level]:
                temp_dict["read_latency_" + level + "_" + level_entry] = std_record.read_latency_map[level][level_entry]

        result_dict_list.append(temp_dict)

    filed_names = temp_dict.keys()

    import csv

    csv_file_name = "csv_results/tradeoff_analysis.csv"
    try:
        with open(csv_file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=filed_names)
            writer.writeheader()
            for data_line in result_dict_list:
                writer.writerow(data_line)
    except IOError:
        print("I/O error")
