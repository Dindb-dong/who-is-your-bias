import json
import argparse
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

torch.manual_seed(42)
weight=torch.randn(6144,2048,dtype=torch.bfloat16)*0.02
raw=torch.randn(768,2048,dtype=torch.bfloat16)
grad=torch.randn(768,6144,dtype=torch.bfloat16)*0.01
parser=argparse.ArgumentParser(description='Bounded frozen BF16 linear cast benchmark; no model weights.')
parser.add_argument('--output',type=Path,required=True)
output=parser.parse_args().output
if output.exists():raise ValueError('Choose a new output file')
output.parent.mkdir(parents=True,exist_ok=True)

def native(x):return F.gelu(F.linear(x,weight))
def cast(x):
    with torch.autocast('cpu',enabled=False):
        return F.gelu(F.linear(x.float(),weight.float()).to(x.dtype))

def run(fn,checkpointed):
    x=raw.detach().clone().requires_grad_(True)
    begin=time.perf_counter()
    y=checkpoint(fn,x,use_reentrant=False) if checkpointed else fn(x)
    y.backward(grad)
    elapsed=time.perf_counter()-begin
    return elapsed,y.detach(),x.grad.detach()

result={'schema_version':'frozen-bf16-linear-cast-probe.v1','shape':[768,2048,6144],
 'torch_version':torch.__version__,'cpu_capability':torch.backends.cpu.get_cpu_capability(),
 'includes':['input cast','weight cast','output cast','GELU','input backward','nonreentrant checkpoint recomputation where declared'],
 'weight_gradient':False,'full_model_loaded':False,'scope':'Toy kernels under possibly concurrent host work; no end-to-end speed claim.','results':[]}
for threads in [4,8]:
    torch.set_num_threads(threads)
    for checkpointed in [False,True]:
        run(native,checkpointed);run(cast,checkpointed)
        times={'native_bf16':[],'fp32_cast':[]}
        values={}
        for repeat in range(3):
            pairs=[('native_bf16',native),('fp32_cast',cast)]
            if repeat%2:pairs.reverse()
            for name,fn in pairs:
                elapsed,y,g=run(fn,checkpointed)
                times[name].append(elapsed);values[name]=(y,g)
        a,b=values['native_bf16'],values['fp32_cast']
        errors={label:{'max_absolute':float((av.float()-bv.float()).abs().max()),
                       'relative_l2':float((av.float()-bv.float()).norm()/av.float().norm())}
                for label,av,bv in [('output',a[0],b[0]),('input_gradient',a[1],b[1])]}
        medians={k:statistics.median(v) for k,v in times.items()}
        row={'threads':threads,'checkpointed':checkpointed,'times_seconds':times,'median_seconds':medians,
             'native_over_cast_speed_ratio':medians['native_bf16']/medians['fp32_cast'],'numerical_difference':errors}
        result['results'].append(row);output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
