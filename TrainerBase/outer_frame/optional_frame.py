import torch
import torch.distributed as dist
from typing import Dict, Any

# from frame_base import FrameBase
from outer_frame.frame_base import get_job_dir
from outer_frame.simple_frame import SimpleFrame

# from old.common_base import CommonBaseDriver
from inner_frame.sampler_base import SamplerBaseDriver2

from Optional_tools.utils_decorator import timer_print
from Optional_tools.utils import setup_rs, report_rs
from Optional_tools.utils import enable_runtime_stats, disable_runtime_stats

# from old.ddp import DDPDriver

# Optional
def consume_prefix_in_state_dict_if_present(
    state_dict: Dict[str, Any], prefix: str
) -> None:
    r"""Strip the prefix in state_dict in place, if any.

    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can
        load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict,
        "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.

    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.

    """

    #state_dict = _state_dict.copy()
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix):]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix):]
            metadata[newkey] = metadata.pop(key)
    return state_dict

# Optional
def update_best_acc(drv: SamplerBaseDriver2, job_dir,
                    acc, best_acc,
                    trial, epoch, delta, best_epoch):
    if acc > best_acc:
        best_acc = acc
        this_epoch = epoch + delta - 1
        best_epoch = this_epoch

        if drv.is_main_proc:
            # Save model
            torch.save(
                drv.model.state_dict(),
                job_dir.joinpath(f'model_{trial}_{this_epoch}.pt'))

            # Write epoch & acc
            with job_dir.joinpath('metadata.txt').open('a') as f:
                f.write(','.join(map(str, (this_epoch, acc))))
                f.write('\n')
        # Point1 ADD: 本来はメインのみ(rank=0)で共有ファルダ（NFSなど）上でセーブし、
        # メイン、サブノード(rank!=0)でこのモデル（ptファイル）をロードする使用になっている。
        # NFS環境がなくてもいいように、サブでも共有ファルダ（NFSなど）上でセーブさせるように修正。
        else:
            torch.save(
                drv.model.state_dict(),
                job_dir.joinpath(f'model_{trial}_{this_epoch}.pt'))

    if drv.is_main_proc:
        print("Best validation accuracy so far: " + str(best_acc))

    if isinstance(drv, SamplerBaseDriver2):
        dist.barrier()

    return best_epoch, best_acc

# Optional
def load_best_model(drv: SamplerBaseDriver2, job_dir, trial, best_epoch):
    if drv.is_main_proc:
        print("\nPerforming inference on trained model at " +
              str(job_dir.joinpath(f'model_{trial}_{best_epoch}.pt')))

    # Optional Best model load
    drv.model.load_state_dict(torch.load(
        job_dir.joinpath(f'model_{trial}_{best_epoch}.pt')))


# Optional
def do_test_run(drv: SamplerBaseDriver2, args, trial_results):
    for i in range(0, len(args.do_test_run_filename)):
        if isinstance(drv, SamplerBaseDriver2):
            drv.model.module.load_state_dict(
                consume_prefix_in_state_dict_if_present(
                    torch.load(args.do_test_run_filename[i]),
                    'module.'))
        else:
            drv.model.load_state_dict(
                consume_prefix_in_state_dict_if_present(
                    torch.load(args.do_test_run_filename[i]),
                    'module.'))
        if isinstance(drv, SamplerBaseDriver2):
            dist.barrier()
        if drv.is_main_proc:
            print("\nPerforming inference on trained model at " +
                  args.do_test_run_filename[i])
        acc = drv.test(('test',))['test']
        if isinstance(drv, SamplerBaseDriver2):
            dist.barrier()
        if drv.is_main_proc:
            print("Final test accuracy is: " + str(acc))
        trial_results.append((acc, args.do_test_run_filename[i]))
        drv.flush_logs()


class OptionalFrame(SimpleFrame):
    def __init__(self, drv_type):
        super().__init__(drv_type=drv_type)

    # Optional
    @timer_print("trial_process", print)
    def trial_process(self, args, drv: SamplerBaseDriver2):
        trial_results = []
        # Optional Trial loop
        for TRIAL in range(0, args.trials):
            drv.reset()

            delta = min(args.test_epoch_frequency, args.epochs)
            do_eval = args.epochs >= args.test_epoch_frequency

            # Optional Best Accuracy
            best_acc = 0
            best_acc_test = 0
            best_epoch = None

            job_dir = get_job_dir(args)

            # Optional Do test run only
            if args.do_test_run:
                do_test_run(drv, args, trial_results)
                break

            # Remove
            if drv.is_main_proc:
                print()
                print("+" + "-" * 40 + "+")
                print("+" + " " * 16 + "TRIAL " + "{:2d}".format(TRIAL) + " " * 16 + "+")
                print("+" + "-" * 40 + "+")

            if drv.is_main_proc:
                print("Performing training")

            for epoch in range(0, args.epochs, delta):

                if isinstance(drv, SamplerBaseDriver2):
                    dist.barrier()

                enable_runtime_stats()
                drv.train(range(epoch, epoch + delta))
                disable_runtime_stats()
                if do_eval:
                    # Driver Test(valid data)
                    acc_type = 'valid'
                    acc = self.test_process(drv, acc_type)
                    if drv.is_main_proc:
                        print("validation accuracy so far: " + str(acc))
                        drv.log((acc_type, 'Accurracy', acc))

                    best_epoch, best_acc = update_best_acc(drv, job_dir, acc, best_acc, TRIAL, epoch, delta, best_epoch)

                drv.flush_logs()

            # Optional
            # report_rs(drv.log)
            report_rs(print)

            # trial_results = finish_process(drv, job_dir, TRIAL, best_epoch, trial_results)
            # Optional
            load_best_model(drv, job_dir, TRIAL, best_epoch)

            # Driver Test(valid & test data)
            acc_dict = self.final_test_process(drv)
            final_valid_acc, final_test_acc = acc_dict['valid'], acc_dict['test']

            # Optional
            trial_results.append((final_valid_acc, final_test_acc))
            if drv.is_main_proc:
                print("\nFinal validation,test accuracy is: " +
                      str(final_valid_acc) + "," + str(final_test_acc) +
                      " on trial " + str(TRIAL))

        if drv.is_main_proc:
            print("")
            drv.log(('End results for all trials', str(trial_results)))

    # Optional タイマー機能を付加
    @timer_print("train_process", print)
    def train_process(self, args, drv: SamplerBaseDriver2):
        if drv.is_main_proc:
            print()
            print("+" * 50)
            print("+" + " " * 16 + "Run main process" + " " * 16 + "+")
            print("+" * 50)

        if drv.is_main_proc:
            print("Performing training")

        super().train_process(args, drv)


    def run_driver(self, args, drv: SamplerBaseDriver2):

        # Optional Runtime statics
        setup_rs(" ".join(["PyGSampler", args.dataset_name, args.model_name]), True)

        if args.use_trial:
            # Optional Trial loop(SALIENTベースのTrialループ処理(Main processを複数回実行する))
            self.trial_process(args, drv)
        else:
            # Train process(シンプルな学習処理にタイマー機能を付加している)
            self.train_process(args, drv)

        drv.flush_logs()


from example.ns_driver import NSDriver
if __name__ == '__main__':
    assert torch.cuda.is_available()
    # Opt_ins = OptionalFrame(drv_type=DDPDriver)
    Opt_ins = OptionalFrame(drv_type=NSDriver)
    Opt_ins.main()
