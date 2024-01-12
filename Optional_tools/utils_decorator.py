import torch
import time


from Optional_tools.utils import Timer, CUDAAggregateTimer
from Optional_tools.utils import setup_runtime_stats, report_runtime_stats
from Optional_tools.utils import enable_runtime_stats, disable_runtime_stats
from Optional_tools.utils import append_runtime_stats, start_runtime_stats_epoch
from Optional_tools.utils import runtime_stats_cuda, is_performance_stats_enabled
from Optional_tools.utils import setup_rs, report_rs, get_rs_status


###########################################################################
# テスト用の出力関数
#
def print_example(x):
    print("Result", x)


###########################################################################
# utils.Timerコンテキストマネージャーとデコレ―タの使用例
#
def timer_print(timer_str, print_func=print):
    def decorator_timer(func):
        def wrapper_function(*args, **kwargs):
            # with構文で利用
            # Timer(出力時のヘッダー文字列, 出力関数)
            with Timer(timer_str, print_func):
                ret = func(*args, **kwargs)
            return ret

        return wrapper_function
    return decorator_timer


def decorator_timer(func):

    def wrapper_function(timer_str, print_func=print, *args, **kwargs):
        # with構文で利用
        # Timer(出力時のヘッダー文字列, 出力関数)
        with Timer(timer_str, print_func):

             ret = func(*args, **kwargs)

        return ret

    return wrapper_function


# コンテキストマネージャーの使用例:
def example_func0(imer_str_arg, print_func, a,b,c=None):
    with Timer(imer_str_arg, print_func):
        print(a,b,c)
# example_func0("example0", print, 1,b=22,c=33)
# result:
# 1 22 33
# example0 took 4.6622e-05 sec

# デコレ―タを使用した例
# @decorator_timer
@timer_print("example1", print)
def example_func1(a,b,c=None):
    print(a,b,c)
# example_func1("example1", print, 1,b=22,c=33)
# example_func1(1,b=22,c=33)

# 1 22 33
# example1 took 5.5104e-05 sec

# コンテキストマネージャーの使用例(出力関数を指定した例):
def example_func2(imer_str_arg, print_func, a,b,c=None):
    with Timer(imer_str_arg, print_func):
        print(a,b,c)
# example_func2("example2", print_example, 1,b=22,c=33)
# result:
# 1 22 33
# Result example2 took 1.0292e-05 sec


# コンテキストマネージャーを使用して、詳細操作する例:
def example_func3(imer_str_arg, print_func, a,b,c=None):
    with Timer((imer_str_arg), print_func) as timer:
        print(a,b,c)
        timer.stop()
        print("Finish!")
# example_func3("example3", print_example, 1,b=22,c=33)
# result:
# 1 22 33
# Finish!
# Result example3 took 3.0288e-05 sec



###########################################################################
# utils.CUDAAggregateTimerのデコレ―タの使用例
#
def cuda_timer_print(timer_str):

    def decorator_cuda_aggregate_timer(func):

        def wrapper_function(*args, **kwargs):

            ctimer = CUDAAggregateTimer(timer_str)
            ctimer.start()

            ret = func(*args, **kwargs)

            ctimer.end()

            return ret

        return wrapper_function

    return decorator_cuda_aggregate_timer



###########################################################################
# utils.RuntimeStatisticsCUDAのデコレ―タとその使用例
#

# RuntimeStatisticsのEpoch開始処理デコレ―タ
# 削除予定
def decorator_rs_start_epoch(func):

    def wrapper_function(*args, **kwargs):
        start_runtime_stats_epoch()

        func(*args, **kwargs)

        return

    return wrapper_function


# RuntimeStatisticsのセットアップ&統計データ出力デコレ―タ
def decorator_rs_setup(func):

    def wrapper_function(print_func, rs_name, performance_stats=True, stats=None,
                         *args, **kwargs):

        # RuntimeStatisticsのセットアップ関数
        # setup_runtime_stats(args)
        setup_rs(rs_name, performance_stats, stats)

        ret = func(*args, **kwargs)

        # RuntimeStatisticsのセットアップ関数
        # print("Final Report:")
        report_rs(print_func)

        return ret

    return wrapper_function


# RuntimeStatisticsのセットアップ&統計データ出力デコレ―タ(引数有り)
def rs_setup(print_func, rs_name, performance_stats=True, stats=None):

    def decorator_rs_setup(func):

        def wrapper_function(*args, **kwargs):
            # RuntimeStatisticsのセットアップ関数
            # setup_runtime_stats(args)
            setup_rs(rs_name, performance_stats, stats)

            ret = func(*args, **kwargs)

            # RuntimeStatisticsのセットアップ関数
            # print("Final Report:")
            report_rs(print_func)

            return ret

        return wrapper_function

    return decorator_rs_setup

# RuntimeStatisticsの統計データ出力可否を設定するデコレ―タ
def decorator_rs_enable(func):
    def wrapper_function(*args, **kwargs):
        enable_runtime_stats()
        ret = func(*args, **kwargs)
        disable_runtime_stats()
        return ret

    return wrapper_function


# RuntimeStatisticsのEpoch開始&終了処理デコレ―タ
def decorator_rs_process_epoch(func):
    assert torch.cuda.is_available()

    def wrapper_function(rs_name=None, *args, **kwargs):
        if rs_name != None:
            runtime_stats_cuda.name = rs_name
        runtime_stats_cuda.start_epoch()

        # status, ret = func(*args, **kwargs)

        ret = func(*args, **kwargs)
        status = get_rs_status()

        # print("Epoch End")
        # Epoch終了処理。全てのデータを統計処理する(平均値、分散を求める)。
        runtime_stats_cuda.end_epoch()
        # 全ての統計データ(平均値、分散)を統計処理する。
        runtime_stats_cuda.report_stats(status)

        return ret


    return wrapper_function

# RuntimeStatisticsのEpoch開始&終了処理デコレ―タ(引数有り)
def rs_process_epoch(rs_name=None):

    def decorator_rs_process_epoch(func):
        # assert torch.cuda.is_available()

        def wrapper_function( *args, **kwargs):
            if rs_name != None:
                runtime_stats_cuda.name = rs_name
            runtime_stats_cuda.start_epoch()

            # status, ret = func(*args, **kwargs)

            ret = func(*args, **kwargs)
            status = get_rs_status()

            # print("Epoch End")
            # Epoch終了処理。全てのデータを統計処理する(平均値、分散を求める)。
            runtime_stats_cuda.end_epoch()
            # 全ての統計データ(平均値、分散)を統計処理する。
            runtime_stats_cuda.report_stats(status)

            return ret

        return wrapper_function

    return decorator_rs_process_epoch


# RuntimeStatisticsのCudaEvnet時刻取得開始&終了処理デコレ―タ
def decorator_rs_process_region(func):
    assert torch.cuda.is_available()

    def wrapper_function(region_name="region(default)", set_st=False, set_en=False,
                         *args, **kwargs):

        # RuntimeStatisticsCUDAが持つ最後に記録されたCudaEvent時刻を利用するかどうかの設定。
        use_last_event_st = runtime_stats_cuda.get_last_event() if set_st else None
        use_last_event_en = runtime_stats_cuda.get_last_event() if set_en else None

        # region_nameは、処理名称を指定(totalやsamplingなど)。
        runtime_stats_cuda.start_region(region_name, use_last_event_st)

        ret = func(*args, **kwargs)

        runtime_stats_cuda.end_region(region_name, use_last_event_en)

        return ret

    return wrapper_function


# RuntimeStatisticsのCudaEvnet時刻取得開始&終了処理デコレ―タ(引数有り)
def rs_process_region(region_name="region(default)", set_st=False, set_en=False,):

    def decorator_rs_process_region(func):
        # assert torch.cuda.is_available()

        def wrapper_function(*args, **kwargs):
            # RuntimeStatisticsCUDAが持つ最後に記録されたCudaEvent時刻を利用するかどうかの設定。
            use_last_event_st = runtime_stats_cuda.get_last_event() if set_st else None
            use_last_event_en = runtime_stats_cuda.get_last_event() if set_en else None

            # region_nameは、処理名称を指定(totalやsamplingなど)。
            runtime_stats_cuda.start_region(region_name, use_last_event_st)

            ret = func(*args, **kwargs)

            runtime_stats_cuda.end_region(region_name, use_last_event_en)

            return ret

        return wrapper_function

    return decorator_rs_process_region


# RuntimeStatisticsのデコレ―タ使用例。
# @decorator_rs_process_epoch
@rs_process_epoch("Example4")
def example_func4(a,b,c=None):

    print(a,b,c)
    # ret5 = example_func5("example5", False, False, 4, b=55, c=666)
    # ret6 = example_func6("example6", True, False, 7, b=88, c=999)

    ret5 = example_func5(4, b=55, c=666)
    ret6 = example_func6(7, b=88, c=999)

    ret = ((a,b,c), ret5, ret6)
    return ret


# @decorator_rs_process_region
@rs_process_region("example5", False, False)
def example_func5(a,b,c=None):
    print(a,b,c)
    ret = (a,b,c)
    # time.sleep(10)
    return ret

# @decorator_rs_process_region
@rs_process_region("example6", True, False)
def example_func6(a,b,c=None):
    print(a,b,c)
    ret = (a,b,c)
    # for _ in range((a+1)*10):
    #     time.sleep(10)
    return ret


# @decorator_rs_setup
# @decorator_rs_enable
# def example_func7():
#
#     ret_gl = example_func4("Example4", 1,b=22,c=333)
#     print(ret_gl)
#     return ret_gl

stats_keys = {'example5': 'Example5', 'example6': 'Example6'}
# @decorator_rs_setup
@rs_setup(print_example, "sampler_model_dataset", True, stats_keys)
@decorator_rs_enable
def example_func7_1():
    for epoch in range(3):
        ret_gl = example_func4(1,b=22,c=333)
        print(ret_gl)
    result = None
    return result

# stats_keys = {'example5': 'Example5', 'example6': 'Example6'}
# example_func7_1(print_example, "sampler_model_dataset", True, stats_keys)
# example_func7_1()
# 出力結果
# 1 22 333
# 4 55 666
# 7 88 999
# +---------------------+--------------------------------+-------+
# | Activity (Example4) | Mean time (ms) (over 0 epochs) | Stdev |
# +---------------------+--------------------------------+-------+
# |       Example5      |              N/A               |  N/A  |
# |       Example6      |              N/A               |  N/A  |
# +---------------------+--------------------------------+-------+
# ((1, 22, 333), (4, 55, 666), (7, 88, 999))
# 1 22 333
# 4 55 666
# 7 88 999
# +---------------------+--------------------------------+-------+
# | Activity (Example4) | Mean time (ms) (over 1 epochs) | Stdev |
# +---------------------+--------------------------------+-------+
# |       Example6      |      0.05728000029921532       |  N/A  |
# |       Example5      |      0.06006399914622307       |  N/A  |
# +---------------------+--------------------------------+-------+
# ((1, 22, 333), (4, 55, 666), (7, 88, 999))
# 1 22 333
# 4 55 666
# 7 88 999
# +---------------------+--------------------------------+----------------------+
# | Activity (Example4) | Mean time (ms) (over 2 epochs) |        Stdev         |
# +---------------------+--------------------------------+----------------------+
# |       Example6      |      0.06367999874055386       | 0.009050964594907632 |
# |       Example5      |      0.06630400009453297       | 0.008824693970320832 |
# +---------------------+--------------------------------+----------------------+
# ((1, 22, 333), (4, 55, 666), (7, 88, 999))
# +---------------------+--------------------------------+----------------------+
# | Activity (Example4) | Mean time (ms) (over 2 epochs) |        Stdev         |
# +---------------------+--------------------------------+----------------------+
# |       Example6      |      0.06367999874055386       | 0.009050964594907632 |
# |       Example5      |      0.06630400009453297       | 0.008824693970320832 |
# +---------------------+--------------------------------+----------------------+
# Result ('performance_breakdown_stats', "[['Example5', 0.06630400009453297, 0.008824693970320832], ['Example6', 0.06367999874055386, 0.009050964594907632]]")


# def decorator_(func):
#     def wrapper_function(*args, **kwargs):
#
#         func(*args, **kwargs)
#
#
#         return
#
#     return wrapper_function

