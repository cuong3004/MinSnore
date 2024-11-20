import os

# Thiết lập biến môi trường
# os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from types import FunctionType
import jax
from flax import linen as nn
from jax import numpy as jnp
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
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
from snoring.utils.module import AudiosetModule, DataIterator
# from snoring.utils.trainer import Trainer
import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
from snoring.dymn_jax.model_jax import get_model
from snoring.config import args
import pytorch_lightning as pl
import torch
from flax.training import train_state

print("Jax device count:", jax.local_device_count())

@partial(jax.jit, static_argnums=(4))
def loss_fn(params, X, Y, state, averaged=True):
    """Defined in :numref:`subsec_layer-normalization-in-bn`"""

    Y_hat, updates = state.apply_fn({'params': params,
                                        'batch_stats': state.batch_stats,
                                        'immutable': state.immutable},
                                    X, mutable=['batch_stats'],
                                    rngs={'dropout': state.dropout_rng})

    fn = optax.sigmoid_binary_cross_entropy

    # Y = jax.nn.one_hot(Y, 10)
    # print(Y)
    return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)


@jax.jit
def make_train_step(params, x, y, state):
    # print(jax.tree_map(jnp.shape, params))
    # print(x.shape)
    (l, mutated_vars), grads = jax.value_and_grad(
            loss_fn, has_aux=True)(params, x, y, state)

    l = jax.lax.pmean(l, 'devices')
    mutated_vars = unfreeze(mutated_vars)
    mutated_vars["batch_stats"] = jax.tree_map(partial(jax.lax.pmean, axis_name='devices'), \
                                                         mutated_vars['batch_stats'])

    grads = jax.lax.pmean(grads, 'devices')
    
    state = state.apply_gradients(grads=grads)
    
    state = state.replace(
            dropout_rng= jax.random.split(state.dropout_rng)[0]
        )
    state = state.replace(batch_stats=mutated_vars['batch_stats'])

    return {"loss": l}, state




class FlaxLightning(pl.LightningModule):
    def __init__(self, lr=0.001, input_shape=(2,1,128,625)):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.model  = get_model(width_mult=0.4)
        
        self.configure_optimizers()
        
        self.global_step_ = 0

    def prepare_batch(self, batch):
        images, labels = batch 
        # images = images.permute(0,3,1,2)
        batch = (images, labels)
        # batch = jax.tree_map(lambda torch_array: torch_array.numpy(), batch)
        return shard(batch)
    
    def configure_optimizers(self):
        self.optim = optax.adam(3e-4)
    
    def on_fit_start(self) -> None:

        root_key = jax.random.PRNGKey(0)
        # print(root_key)
        print("start fit")
        params_key, dropout_key = jax.random.split(root_key)
        key = {'params': params_key, 'dropout': dropout_key}
        # print("OK?")
        dummy_input = jnp.ones(self.hparams.input_shape, dtype=args['input_dtype'])
        
        variables = self.model.init(key, dummy_input)
        variables = jax.tree_map(lambda x : x.astype(args['input_dtype']), variables)
        params = variables['params']
        batch_stats = variables['batch_stats']
        immutable = variables["immutable"]
        print("OK output")
        num_params = sum(p.size for p in jax.tree_leaves(params))
        print("Number of parameters:", num_params)
        

        # print(batch_stats)

        class TrainState(train_state.TrainState):
            batch_stats: Any
            dropout_rng: jax.Array
            immutable: Any

        state = TrainState.create(apply_fn=self.model.apply,
                                       params=params,
                                       batch_stats=batch_stats,
                                       dropout_rng=dropout_key,
                                       immutable=immutable,
                                       tx=self.optim)

        pl_state = flax.jax_utils.replicate(state)
        pl_state = pl_state.replace(dropout_rng=shard_prng_key(dropout_key))

        self.state = pl_state


        self.pl_training_step = jax.pmap(make_train_step, \
                                        axis_name='devices')
        
    def training_step(self, batch, batch_idx):
        batch = self.prepare_batch(batch)
        images, labels = batch
        # print(images.shape)
        # print(labels.shape)
        
        pl_metrics, self.state = self.pl_training_step(self.state.params,
                                                               images,
                                                               labels,
                                                               self.state)
        
        
        # print("out grad")
        # print(jax.tree_map(jnp.shape, grads))
        # self.state = self.state.apply_gradients(grads=grads)
        # print("apply grad")
        # self.state = self.state.replace(
            # dropout_rng= jax.pmap(lambda x: jax.random.split(x)[0])(self.state.dropout_rng)
        # )
        # self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])

        metrics = flax.jax_utils.unreplicate(pl_metrics)
        
        dict = jax.tree_map(lambda x: torch.scalar_tensor(x.item()), metrics)
        self.log_dict(dict, prog_bar=True)
        self.global_step_ += 1
        return dict
    
    def on_fit_end(self):
        pass
        # pathlib.Path.mkdir(phd_path / 'checkpoints/ScoreBased', parents=True, exist_ok=True)
        # eqx.tree_serialise_leaves(phd_path / f"checkpoints/ScoreBased/last.eqx", self.model)

    def on_train_epoch_end(self) -> None:
        pass
        # pathlib.Path.mkdir(phd_path / 'checkpoints/ScoreBased', parents=True, exist_ok=True)
        # eqx.tree_serialise_leaves(phd_path / f"checkpoints/ScoreBased/last.eqx", self.model)
    
    



# class LeNet(Module):  #@save
#     """The LeNet-5 model."""
#     lr: float = 0.1
#     num_classes: int = 527
#     kernel_init: FunctionType = nn.initializers.xavier_uniform
#     training: bool = False

#     def setup(self):
#         self.net = get_model(width_mult=0.4)
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
    
    # @partial(jax.jit, static_argnums=(0, 5))
    # def loss(self, params, X, Y, state, averaged=True):
    #     """Defined in :numref:`subsec_layer-normalization-in-bn`"""
    #     Y_hat, updates = state.apply_fn({'params': params,
    #                                      'batch_stats': state.batch_stats},
    #                                     *X, mutable=['batch_stats'],
    #                                     rngs={'dropout': state.dropout_rng})
        
    #     fn = optax.sigmoid_binary_cross_entropy
    #     return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)
    
    # @partial(jax.jit, static_argnums=(0, 5))
    # def accuracy_train(self, params, X, Y, state, averaged=True):
    #     Y_hat, _ = state.apply_fn(
    #         {'params': params,
    #         'batch_stats': state.batch_stats},  # BatchNorm Only 
    #         *X, mutable=['batch_stats'])
            
    #     preds = jnp.where(Y_hat > 0.5, 1, 0)
    #     compare = jnp.astype(preds == Y, jnp.float32)
    #     return jnp.mean(compare) if averaged else compare

    # @partial(jax.jit, static_argnums=(0, 5))
    # def accuracy_val(self, params, X, Y, state, averaged=True):
    #     Y_hat = state.apply_fn(
    #         {'params': params,
    #         'batch_stats': state.batch_stats},  # BatchNorm Only
    #         *X)

    #     preds = jnp.where(Y_hat > 0.5, 1, 0)
    #     compare = jnp.astype(preds == Y, jnp.float32)
    #     return jnp.mean(compare) if averaged else compare
    
    # def training_step(self, params, batch, state):
    #     print("begin_train")
    #     (l, mutated_vars), grads = jax.value_and_grad(
    #         self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        
    #     l = jax.lax.pmean(l, 'devices')
    #     mutated_vars = unfreeze(mutated_vars)
    #     mutated_vars["batch_stats"] = jax.tree_map(functools.partial(lax.pmean, axis_name='devices'), \
    #                                                      mutated_vars['batch_stats'])
    #     grads = lax.pmean(grads, 'devices')
    
    #     acc = self.accuracy_train(params, batch[:-1], batch[-1], state)
    #     acc = lax.pmean(acc, 'devices')
        
    #     return ({"loss": l, "accuracy":acc}, mutated_vars), grads

    # def validation_step(self, params, batch, state):
    #     print("valid")
    #     l, _ = self.loss(params, batch[:-1], batch[-1], state)
    #     l = jax.lax.pmean(l, 'devices')
        
    #     acc = self.accuracy_val(params, batch[:-1], batch[-1], state)
    #     acc = lax.pmean(acc, 'devices')
        
    #     return {"loss": l, "accuracy":acc}
    
    # def configure_optimizers(self):
    #     """Defined in :numref:`sec_classification`"""
    #     return optax.sgd(self.lr)


from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
# from snoring.utils.common import ProgressBoard

# LeNet().init(jax.random.PRNGKey(0), jnp.ones([128,28,28,1]))
# WandbLogger(project="MNIST") 
wandb_logger = WandbLogger(project="MNIST") 


# board = ProgressBoard(wandb_logger)


modelmodule = FlaxLightning(1e-4)

trainer = pl.Trainer(max_epochs=15, devices=1, accelerator="cpu", logger=wandb_logger)
data = AudiosetModule(args)
train_dataloader = DataIterator(data.train_dataloader(), 1000)

trainer.fit(modelmodule, train_dataloader)