import torch
import torch.distributed.rpc as rpc



# Pytorch sample code
# memo
#
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


# memo
#
def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


def remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), call_method, args=args, kwargs=kwargs)


