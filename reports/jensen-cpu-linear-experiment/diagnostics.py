import copy
import json
import argparse
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from peft.tuners.lora.layer import Linear as LoraLinear

parser=argparse.ArgumentParser(description='Tiny LoRA and saved-tensor diagnostics; no model weights.')
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output
if out.exists():raise ValueError('Choose a new output file')
out.parent.mkdir(parents=True,exist_ok=True)

torch.set_num_threads(2)
torch.manual_seed(77)


def cast_forward(self,x):
    assert not self.weight.requires_grad and self.weight.dtype==x.dtype==torch.bfloat16
    with torch.autocast('cpu',enabled=False):
        return F.linear(x.float(),self.weight.float(),None).to(x.dtype)


class TinyLora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        for name,ni,no in [('first',128,192),('second',192,128)]:
            base=torch.nn.Linear(ni,no,bias=False,dtype=torch.bfloat16)
            base.weight.requires_grad_(False)
            layer=LoraLinear(base,'default',r=4,lora_alpha=8,lora_dropout=.05)
            layer.lora_A['default'].to(torch.float32)
            layer.lora_B['default'].to(torch.float32)
            setattr(self,name,layer)
    def forward(self,x):return self.second(F.gelu(self.first(x)))


native=TinyLora();candidate=copy.deepcopy(native)
for layer in [candidate.first,candidate.second]:
    layer.base_layer.forward=MethodType(cast_forward,layer.base_layer)
assert native.state_dict().keys()==candidate.state_dict().keys()
original={name:p.detach().clone() for name,p in native.named_parameters() if not p.requires_grad}
values=torch.randn(2,8,128,dtype=torch.bfloat16)
target=torch.randn(2,8,128)
losses={}
for name,model in [('native_bf16',native),('fp32_cast',candidate)]:
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002)
    loss_values=[]
    for step in range(4):
        torch.manual_seed(1000+step)
        optimizer.zero_grad()
        x=values.clone().requires_grad_(True)
        prediction=checkpoint(model,x,use_reentrant=False)
        loss=F.mse_loss(prediction.float(),target)
        loss.backward();optimizer.step();loss_values.append(float(loss.detach()))
        assert x.grad is not None and torch.isfinite(x.grad).all()
    losses[name]=loss_values
    for pname,p in model.named_parameters():
        if not p.requires_grad:
            assert p.grad is None and torch.equal(p,original[pname])
        else:
            assert p.dtype==torch.float32 and torch.isfinite(p).all()
differences={}
for (name,a),(other,b) in zip(native.named_parameters(),candidate.named_parameters(),strict=True):
    assert name==other
    if a.requires_grad:
        torch.testing.assert_close(a,b,rtol=.02,atol=2e-5)
        differences[name]={'max_absolute':float((a-b).abs().max().detach()),'relative_l2':float(((a-b).norm()/a.norm()).detach())}
assert max(abs(a-b) for a,b in zip(losses['native_bf16'],losses['fp32_cast']))<1e-4

# Observe retained tensors at the end of forward; this is not process peak RSS.
weight=torch.randn(6144,2048,dtype=torch.bfloat16)*.02
raw=torch.randn(768,2048,dtype=torch.bfloat16)
def normal(x):return F.gelu(F.linear(x,weight))
def promoted(x):
    with torch.autocast('cpu',enabled=False):return F.gelu(F.linear(x.float(),weight.float()).to(x.dtype))
retention=[]
for checkpointed in [False,True]:
    for name,fn in [('native_bf16',normal),('fp32_cast',promoted)]:
        records=[]
        def pack(tensor):
            records.append({'shape':list(tensor.shape),'dtype':str(tensor.dtype),'bytes':tensor.numel()*tensor.element_size(),
                'aliases_existing_frozen_weight':tensor.untyped_storage().data_ptr()==weight.untyped_storage().data_ptr()})
            return tensor
        x=raw.clone().requires_grad_(True)
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
            result=checkpoint(fn,x,use_reentrant=False) if checkpointed else fn(x)
        retention.append({'condition':name,'checkpointed':checkpointed,'saved_tensors':records,
                          'saved_float32_bytes':sum(r['bytes'] for r in records if r['dtype']=='torch.float32')})
        del result,x

report={'schema_version':'frozen-bf16-linear-cast-diagnostics.v1','full_model_loaded':False,
 'tiny_lora':{'layers':2,'rank':4,'alpha':8,'dropout':.05,'optimizer':'AdamW','learning_rate':.0002,'updates':4,
     'checkpoint_use_reentrant':False,'trainable_dtype':'float32','frozen_dtype':'bfloat16','losses':losses,
     'parameter_comparison_tolerance':{'rtol':.02,'atol':2e-5},'loss_absolute_tolerance':1e-4,
     'trainable_parameter_differences':differences,'frozen_parameters_unchanged':True,'state_dict_keys_unchanged':True,
     'finite_input_gradients':True,'passed':True},
 'saved_tensor_observations':retention,
 'memory_limitations':['FP32 weight casts can remain alive for input backward even though weights are frozen.','Nonreentrant checkpointing avoids retaining these casts between block forward and backward in this toy probe; backward still allocates temporary FP32 weights/input/output.','Saved-tensor bytes are not peak process memory; GEMM workspace, allocator behavior, other layers and output logits are not measured.','A blanket wrapper over an uncheckpointed large LM head could add a substantial FP32 weight/logit allocation.'],
 'decision':'reject_integration_no_material_checkpointed_speed_gain'}

out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
