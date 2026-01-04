from .DDU import DDU
from .MST_Plus_Plus import MST_Plus_Plus, MST_Plus_Plus2
from .MIRNetV2 import MIRNet_v2 as MIRNetV2
from .MIRNetV2 import MIRNet_v2m as MIRNetV2m

def network_generator(method, in_dim, out_dim, deblur, dim, stage, reuse, use_alpha, use_sigma):
    method = method.lower()
    if method == 'mirnetv2':
        return MIRNetV2(in_dim, out_dim, dim, n_RRG=stage)
    elif method == 'mirnetv2m':
        return MIRNetV2m(in_dim, out_dim, dim, n_RRG=stage)
    elif method == 'ddu':
        return DDU(out_dim, deblur, reuse=reuse, use_alpha=use_alpha, use_sigma=use_sigma)
    elif method == 'mst++':
        return MST_Plus_Plus(in_dim, out_dim, dim, stage)
    elif method == 'mst++2':
        return MST_Plus_Plus2(in_dim, out_dim, dim, stage)
    else:
        raise ValueError(f"method {method} not supported")