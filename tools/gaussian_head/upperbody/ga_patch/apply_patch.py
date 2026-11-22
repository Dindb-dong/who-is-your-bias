"""Install the upper-body model into a GaussianAvatars checkout (idempotent).

    python apply_patch.py <GaussianAvatars dir>

Copies smplx_gaussian_model.py and body_model.py into scene/ and makes train.py use SmplxGaussianModel when the
GA_UPPERBODY environment variable points at a smplx_head.npz transplant. Without GA_UPPERBODY nothing changes.
"""
import os
import shutil
import sys

ga = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
shutil.copy(os.path.join(here, 'smplx_gaussian_model.py'), os.path.join(ga, 'scene', 'smplx_gaussian_model.py'))
shutil.copy(os.path.join(here, '..', 'body_model.py'), os.path.join(ga, 'scene', 'body_model.py'))

p = os.path.join(ga, 'train.py')
src = open(p).read()
old = '        gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)\n'
new = ("        if os.environ.get('GA_UPPERBODY'):  # upper-body patch\n"
       "            from scene.smplx_gaussian_model import SmplxGaussianModel\n"
       "            gaussians = SmplxGaussianModel(dataset.sh_degree, os.environ['GA_UPPERBODY'],\n"
       "                                           os.path.join(dataset.source_path, 'smplx_body_param.npz'), dataset.not_finetune_flame_params)\n"
       "        else:\n"
       "            gaussians = FlameGaussianModel(dataset.sh_degree, dataset.disable_flame_static_offset, dataset.not_finetune_flame_params)\n")
if '# upper-body patch' in src:
    print('train.py already patched')
else:
    assert src.count(old) == 1, 'unexpected train.py: model construction line not found'
    open(p, 'w').write(src.replace(old, new))
    print('patched train.py')

# render.py builds the model with its own (shorter) FlameGaussianModel(...) call; switch it the same way.
import re  # noqa: E402
p = os.path.join(ga, 'render.py')
src = open(p).read()
if '# upper-body patch' in src:
    print('render.py already patched')
else:
    pattern = re.compile(r'^(?P<indent>[ \t]*)gaussians = FlameGaussianModel\((?P<args>[^\n]*)\)[ \t]*$', re.M)
    matches = pattern.findall(src)
    assert len(matches) == 1, f'unexpected render.py: {len(matches)} FlameGaussianModel construction lines'

    def switch(m):
        i = m['indent']
        return (f"{i}if os.environ.get('GA_UPPERBODY'):  # upper-body patch\n"
                f"{i}    from scene.smplx_gaussian_model import SmplxGaussianModel\n"
                f"{i}    gaussians = SmplxGaussianModel(dataset.sh_degree, os.environ['GA_UPPERBODY'],\n"
                f"{i}                                   os.path.join(dataset.source_path, 'smplx_body_param.npz'))\n"
                f"{i}else:\n"
                f"{i}    gaussians = FlameGaussianModel({m['args']})")
    src = pattern.sub(switch, src)
    if not re.search(r'^import os\b', src, re.M):
        src = 'import os\n' + src
    open(p, 'w').write(src)
    print('patched render.py')
print('PATCH OK')
