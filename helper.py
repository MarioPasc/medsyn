import numpy as np, pathlib as p
z=np.load(p.Path("/media/mpascual/PortableSSD/medsyn/PathMNIST/PathMNIST.npz"))
for k in sorted(z.keys()): 
    a=z[k]; print(k, a.shape, a.dtype, ("HWC","CHW")[a.ndim==4 and a.shape[1] in (1,3)])
print("train labels unique:", np.unique(z["train_labels"]))
print("synth ratio train:", z["train_is_synth"].mean())
