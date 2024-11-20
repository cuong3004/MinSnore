import inspect
import collections
from dataclasses import field
# from flax import linen as nn
import functools
import jax 
from jax import numpy as jnp


show_shape = functools.partial(jax.tree_map, jnp.shape)


class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
            
            

class ProgressBoard(HyperParameters):
    """The board that plots data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, logger):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
#             self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
#             self.data[label] = []
        points = self.raw_points[label]
#         line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return
        
        mean = lambda x: sum(x) / len(x)
        p = Point(mean([p.x for p in points]),
                          mean([p.y for p in points]))
        points.clear()
        
        self.logger.log_metrics({label: p.y}, step=p.x)