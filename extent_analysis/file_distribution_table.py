
import sys
import os
import pandas as pd

TOTAL_FILE_LENGTH = 6


def num_to_file(file_num_str):
    prefix_length = 6 - len(file_num_str)
    sst_name = "0"*prefix_length+file_num_str+".sst"
    return sst_name


def get_file_info(file_name):
    file_size_result = os.popen("ls -l " + file_name).read()
    if len(file_size_result) <= 5:
        return None, None

    file_frag_result = os.popen("filefrag -v " + file_name).read()
    size_byte = file_size_result.split(" ")[4]
    size_byte = int(size_byte)

    num_extents = file_frag_result.split("\n")[-2].split(" ")[1]

    return size_byte, num_extents


if __name__ == "__main__":
    work_path = "./"
    if len(sys.argv) >= 2:
        work_path = sys.argv[-1]

    std_file = open(os.path.abspath(work_path)+"/stdout.txt")

    files = {}
    level = 0
    for line in std_file.readlines():
        file_names = []
        if "Files in Level" in line:
            file_nums = line.split(" ")[0]
            file_nums = int(file_nums)
            file_num_list = line.split(" ")[-file_nums:-1]

            for file_num in file_num_list:
                file_names.append(file_num)

            files[level] = file_names
            level += 1

    print(files)

    file_info_list = []

    for level in files:
        for file_num in files[level]:
            size, extents = get_file_info(num_to_file(file_num))
            if size:
                file_info = [level, file_num, size, extents]
                file_info_list.append(file_info)

    df = pd.DataFrame(file_info_list, columns=[
                      "level", "file_num", "size_byte", "extents"])
    df.to_csv("extents_file.csv", index=False)
