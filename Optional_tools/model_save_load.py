import torch

from Optional_tools import utils_decorator

# 通常用
def save_model(model_dict, file_path):
    torch.save(model_dict, file_path)

# 通常用
def load_model(file_path):
    torch.load(file_path)


# デバッグ用
def save_model_deco(model_dict, file_path, timer_str=None, print_func=print):
    @utils_decorator.decorator_timer
    def save_model_deco_inner(model_dict_in, file_path_in):
        save_model(model_dict_in, file_path_in)

    timer_str = "".format("save_model. path=%s" %file_path) if timer_str is None else timer_str

    save_model_deco_inner(timer_str, print_func, model_dict, file_path)

# デバッグ用
def load_model_deco(file_path, timer_str=None, print_func=print):
    @utils_decorator.decorator_timer
    def load_model_deco_inner(file_path_in):
        load_model(file_path_in)

    timer_str = "".format("load_model. path=%s" %file_path) if timer_str is None else timer_str

    load_model_deco_inner(timer_str, print_func, file_path)

