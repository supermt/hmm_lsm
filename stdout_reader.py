import re

BENCHMARK_RESULT_REGEX = r"[a-zA-z]+\s+:\s+[0-9]+\.[0-9]+\smicros\/op\s[0-9]+\sops\/sec;\s+.+"

source_file = "trade_off_analysis/StorageMaterial.NVMeSSD/1CPU/64MB/stdout.txt_6241"

COMPACTION_STAT_METRICS = ["Flush(GB)", "Cumulative compaction", "Interval compaction", "Stalls(count)"]

DB_STAT_METRICS = ["Cumulative writes", "Interval writes"]


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
    total_wirte_size = ingest_metrics[0].split(" ")[0]
    return {"total_write_op": total_write_op, "total_write_size": total_wirte_size}


def get_compaction_stat_entry(line, entry_name):
    line = line.replace(entry_name + ":", "")
    if entry_name == "Flush(GB)":
        return {entry_name: line.split(" ")[2]}
    elif entry_name == "Stalls(count)":
        print(entry_name)
    return {"1":"2"}


class stdout_reader():
    def line_markers(self):
        marked_lines = ["benchmarks", "compaction_stat", "file_reading", "db_stats",
                        "statistics"]
        self.line_map = {x: [] for x in marked_lines}

        line_counter = 0
        for line in self.filelines:
            matchObj_benchmark = re.match(BENCHMARK_RESULT_REGEX, line)

            if matchObj_benchmark:
                self.line_map["benchmarks"].append(line_counter)
                self.benchmark_results.update(get_benchmark_entry(line))

            for compaction_metrics in COMPACTION_STAT_METRICS:
                if compaction_metrics in line:
                    self.line_map["compaction_stat"].append(line_counter)
                    self.tradeoff_data.update(get_compaction_stat_entry(line, compaction_metrics))

            for db_metrics in DB_STAT_METRICS:
                if db_metrics in line:
                    self.line_map["db_stats"].append(line_counter)
                    self.tradeoff_data.update(get_db_stat_entry(line))
            line_counter += 1
        pass

    def extract_benchmark_results(self):
        for test_line in self.filelines:
            result_pack = get_benchmark_entry(test_line)
            if result_pack:
                self.benchmark_results.update(result_pack)
            else:
                return

    def __init__(self, input_file):
        self.stdout_file = input_file
        self.filelines = [x.replace("\n", "") for x in open(input_file, "r").readlines()]
        self.benchmark_results = {}
        self.tradeoff_data = {}
        # add basic information
        base_info = self.stdout_file.split("/")
        self.cpu_count = base_info[-3]
        self.batch_size = base_info[-2]
        self.device = base_info[-4]
        # handle the file
        self.line_markers()
        print(self.line_map)

        return


if __name__ == '__main__':
    result_map = {}
    tradeoff_data = []
    stdout_reader(source_file)
    # handle_stdout_file(stdout_file,result_map,tradeoff_data)
