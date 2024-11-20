from snoring.utils.common import HyperParameters
from flax.training import train_state
import jax 
import flax  
from flax.training.common_utils import shard, shard_prng_key
from flax.core import freeze, unfreeze
from jax import numpy as jnp
from typing import Any
from tqdm import tqdm

class Trainer(HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, max_epochs, board, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = 200#len(self.train_dataloader)
        self.num_val_batches = 0 #(len(self.val_dataloader)
                                #if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        self.model = model
        self.model.board = self.board
        
        self.model.board.logger.log_hyperparams({})

    def fit(self, model, data, key=None):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()

        if key is None:
            root_key = jax.random.PRNGKey(0)
        else:
            root_key = key
        params_key, dropout_key = jax.random.split(root_key)
        key = {'params': params_key, 'dropout': dropout_key}

        dummy_input = next(self.train_dataloader)[:-1]
#         print(dummy_input.shape)
        variables = model.apply_init(dummy_input, key=key)
        params = variables['params']

        if 'batch_stats' in variables.keys():
            # Here batch_stats will be used later (e.g., for batch norm)
            batch_stats = variables['batch_stats']
        else:
            batch_stats = {}

        # Flax uses optax under the hood for a single state obj TrainState.
        # More will be discussed later in the dropout and batch
        # normalization section
        class TrainState(train_state.TrainState):
            batch_stats: Any
            dropout_rng: jax.random.PRNGKeyArray

        state = TrainState.create(apply_fn=model.apply,
                                       params=params,
                                       batch_stats=batch_stats,
                                       dropout_rng=dropout_key,
                                       tx=model.configure_optimizers())
        
#         print(show_shape(state))
        pl_state = flax.jax_utils.replicate(state)
#         print(pl_state.dropout_rng)
        pl_state = pl_state.replace(dropout_rng=shard_prng_key(dropout_key))
#         print(pl_state.dropout_rng)
#         print(show_shape(pl_state))
        
#         print(jax.random.split(dropout_key, num=jax.local_device_count()))
        
        self.state = pl_state
        
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        
        self.model.pl_training_step = jax.pmap(self.model.training_step, \
                                                     axis_name='devices')
        self.model.pl_validation_step = jax.pmap(self.model.validation_step, \
                                                    axis_name='devices')
        
        for self.epoch in range(self.max_epochs):
            print(f"[{self.epoch}/{self.max_epochs}]")
            self.fit_epoch()

    def fit_epoch(self):
        """Defined in :numref:`sec_linear_scratch`"""
        self.model.training = True
        self.state = unfreeze(self.state)
        if self.state.batch_stats:
            # Mutable states will be used later (e.g., for batch norm)
            for batch in tqdm(self.train_dataloader, total=self.num_train_batches):                
                (pl_metrics, mutated_vars), grads = self.model.pl_training_step(self.state.params,
                                                               self.prepare_batch(batch),
                                                               self.state)
                
                # print(pl_metrics)
                
                self.state = self.state.apply_gradients(grads=grads)
                # Can be ignored for models without Dropout Layers
                self.state = self.state.replace(
                    dropout_rng= jax.pmap(lambda x: jax.random.split(x)[0])(self.state.dropout_rng)
                )
                self.state = self.state.replace(batch_stats=mutated_vars['batch_stats'])
                
                metrics = flax.jax_utils.unreplicate(pl_metrics)
                for k, v in metrics.items():
                    self.model.plot(k, v, train=True)
                    
                self.train_batch_idx += 1
                
                if self.train_batch_idx % self.num_train_batches == 0:
                    break
        else:
            assert False
        
        self.state = self.state
        
        if self.val_dataloader is None:
            return
        self.model.training = False
        
        for batch in self.val_dataloader:
            pl_metrics = self.model.pl_validation_step(self.state.params,
                                       self.prepare_batch(batch),
                                       self.state)
            
            metrics = flax.jax_utils.unreplicate(pl_metrics)
            for k, v in metrics.items():
                self.model.plot(k, v, train=False)
            
            
            self.val_batch_idx += 1


    def prepare_batch(self, batch):
        """Defined in :numref:`sec_use_gpu`"""
#         if self.gpus:
#             batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return shard(batch)

    def clip_gradients(self, grad_clip_val, grads):
        """Defined in :numref:`sec_rnn-scratch`"""
        grad_leaves, _ = jax.tree_util.tree_flatten(grads)
        norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
        clip = lambda grad: jnp.where(norm < grad_clip_val,
                                      grad, grad * (grad_clip_val / norm))
        return jax.tree_util.tree_map(clip, grads)