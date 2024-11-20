import os

# Thiết lập biến môi trường
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from types import FunctionType
import jax
from flax import linen as nn
from jax import numpy as jnp
# from d2l import jax as d2l
from dataclasses import field
from functools import partial
from types import FunctionType
from typing import Any
# import inspect
# import tensorflow as tf
import tensorflow as tf
tf.config.experimental.set_visible_devices([], 'GPU')
# import tensorflow_datasets as tfds
import optax
# from flax.training import train_state
from flax.training.common_utils import shard, shard_prng_key
import flax
import functools
from jax import lax
from flax.core import freeze, unfreeze
# import collections
from snoring.utils.module import Module, FashionMNIST, AudiosetModule
from snoring.utils.trainer import Trainer
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
from snoring.dymn_jax.model_jax import get_model
from snoring.config import args

print("Jax device count:", jax.local_device_count())

class LeNet(Module):  #@save
    """The LeNet-5 model."""
    lr: float = 0.1
    num_classes: int = 527
    kernel_init: FunctionType = nn.initializers.xavier_uniform
    training: bool = False

    def setup(self):
        self.net = get_model(width_mult=0.4)
        # self.net = nn.Sequential([
        #     nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
        #             kernel_init=self.kernel_init()),
        #     nn.sigmoid,
        #     lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
        #     nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
        #             kernel_init=self.kernel_init()),
        #     nn.BatchNorm(not self.training),
        #     nn.sigmoid,
        #     lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
        #     nn.Conv(features=32, kernel_size=(5, 5), padding='VALID',
        #             kernel_init=self.kernel_init()),
        #     nn.BatchNorm(not self.training),
        #     nn.sigmoid,
        #     lambda x: nn.avg_pool(x, window_shape=(2, 3), strides=(2, 3)),
        #     nn.Conv(features=64, kernel_size=(5, 5), padding='VALID',
        #             kernel_init=self.kernel_init()),
        #     nn.sigmoid,
        #     lambda x: nn.avg_pool(x, window_shape=(2, 3), strides=(2, 3)),
        #     lambda x: x.reshape((x.shape[0], -1)),  # flatten
        #     nn.Dense(features=120, kernel_init=self.kernel_init()),
        #     nn.sigmoid,
        #     nn.Dense(features=84, kernel_init=self.kernel_init()),
        #     nn.sigmoid,
        #     nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        # ])
    
    @partial(jax.jit, static_argnums=(0, 5))
    def loss(self, params, X, Y, state, averaged=True):
        """Defined in :numref:`subsec_layer-normalization-in-bn`"""
        Y_hat, updates = state.apply_fn({'params': params,
                                         'batch_stats': state.batch_stats},
                                        *X, mutable=['batch_stats'],
                                        rngs={'dropout': state.dropout_rng})
        
        fn = optax.sigmoid_binary_cross_entropy
        return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
    
    @partial(jax.jit, static_argnums=(0, 5))
    def accuracy_train(self, params, X, Y, state, averaged=True):
        Y_hat, _ = state.apply_fn(
            {'params': params,
            'batch_stats': state.batch_stats},  # BatchNorm Only 
            *X, mutable=['batch_stats'])
            
        preds = jnp.where(Y_hat > 0.5, 1, 0)
        compare = jnp.astype(preds == Y, jnp.float32)
        return jnp.mean(compare) if averaged else compare

    @partial(jax.jit, static_argnums=(0, 5))
    def accuracy_val(self, params, X, Y, state, averaged=True):
        Y_hat = state.apply_fn(
            {'params': params,
            'batch_stats': state.batch_stats},  # BatchNorm Only
            *X)

        preds = jnp.where(Y_hat > 0.5, 1, 0)
        compare = jnp.astype(preds == Y, jnp.float32)
        return jnp.mean(compare) if averaged else compare
    
    def training_step(self, params, batch, state):
        print("begin_train")
        (l, mutated_vars), grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        
        l = jax.lax.pmean(l, 'devices')
        mutated_vars = unfreeze(mutated_vars)
        mutated_vars["batch_stats"] = jax.tree_map(functools.partial(lax.pmean, axis_name='devices'), \
                                                         mutated_vars['batch_stats'])
        grads = lax.pmean(grads, 'devices')
    
        acc = self.accuracy_train(params, batch[:-1], batch[-1], state)
        acc = lax.pmean(acc, 'devices')
        
        return ({"loss": l, "accuracy":acc}, mutated_vars), grads

    def validation_step(self, params, batch, state):
        print("valid")
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        l = jax.lax.pmean(l, 'devices')
        
        acc = self.accuracy_val(params, batch[:-1], batch[-1], state)
        acc = lax.pmean(acc, 'devices')
        
        return {"loss": l, "accuracy":acc}
    
    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return optax.sgd(self.lr)


from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from snoring.utils.common import ProgressBoard

# LeNet().init(jax.random.PRNGKey(0), jnp.ones([128,28,28,1]))
# WandbLogger(project="MNIST") 
wandb_logger = TensorBoardLogger("tb_logs", name="my_model") #


board = ProgressBoard(wandb_logger)



trainer = Trainer(max_epochs=10, board=board)
data = AudiosetModule(args)
model = LeNet(lr=0.1)
trainer.fit(model, data)