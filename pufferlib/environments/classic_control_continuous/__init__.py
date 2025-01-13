from .environment import env_creator, make

try:
    import torch
except ImportError:
    pass
else:
    from .torch import ContinuousCartPolePolicy as Policy
