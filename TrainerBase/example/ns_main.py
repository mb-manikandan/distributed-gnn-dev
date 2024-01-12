import torch

# from frame_base import FrameBase

from TrainerBase.outer_frame.simple_frame import SimpleFrame
# from SamplerBase.outer_frame.optional_frame import OptionalFrame

from TrainerBase.example.ns_trainer import NSTrainerBase


def simple_main():
    simple_ins = SimpleFrame(drv_type=NSTrainerBase)

    simple_ins.main()


# def optional_main():
#     optional_ins = OptionalFrame(drv_type=NSTrainer)
#     optional_ins.main()


if __name__ == '__main__':
    # TODO: Remove
    # assert torch.cuda.is_available()
    simple_main()
    # optional_main()
