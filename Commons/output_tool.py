from pathlib import Path
from datetime import datetime

def write_func(out_str, file_path):
    with file_path.open('a') as f:
        f.writelines(out_str + '\n')


def output_func(logger_mode=False, out_str="", file_path=Path("./")):
    if logger_mode:
        write_func(out_str, file_path)
    else:
        print(out_str)

