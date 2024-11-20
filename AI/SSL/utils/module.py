from dataclasses import field
from snoring.utils.common import HyperParameters
import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
from jax import numpy as jnp
import jax
from flax.training.common_utils import shard
# import tensorflow_io as tfio
from typing import Iterable

from augmentations import crop_resize, mix_up, random_linear_fader

class DataModule(HyperParameters):
    """The base class of data.

    Defined in :numref:`subsec_oo-design-models`"""
    def __init__(self, root='../data'):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
    
    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        """Defined in :numref:`sec_synthetic-regression-data`"""
        tensors = tuple(a[indices] for a in tensors)
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer).batch(self.batch_size))


class Module(nn.Module, HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
    # No need for save_hyperparam when using Python dataclass
    plot_train_per_epoch: int = field(default=2, init=False)
    plot_valid_per_epoch: int = field(default=1, init=False)
    # Use default_factory to make sure new plots are generated on each run

    def forward(self, X, *args, **kwargs):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X, *args, **kwargs)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        
        self.board.draw(x, value,
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, params, batch, state):
        raise NotImplementedError

    def validation_step(self, params, batch, state):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def apply_init(self, dummy_input, key):
        """Defined in :numref:`sec_lazy_init`"""
        params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
        return params
    




class FashionMNIST(DataModule):
    """The Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()

    def text_labels(self, indices):
        """Return text labels.
    
        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        """Defined in :numref:`sec_fashion_mnist`"""
        data = self.train if train else self.val
        process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                                tf.cast(y, dtype='int32'))
        resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
        shuffle_buf = len(data[0]) if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).batch(
                self.batch_size).map(resize_fn).shuffle(shuffle_buf))
        

class AudiosetModule(DataModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dtype = jnp.bfloat16 if args['use_tpu'] else jnp.float32
        self.label_dtype = self.input_dtype
        self.depth = 527
        self.target_length = 32000*10
        
        self.train_filenames = []
        for train_dir in args['train_dirs']:
            self.train_filenames.extend(tf.io.gfile.glob(train_dir+'/*.tfrec*'))
        # val_filenames = tf.io.gfile.glob(args['val_dir']+'/*.tfrec*')
        # test_filenames = tf.io.gfile.glob(args['test_dir']+'/*.tfrec*')
    
    def read_labeled_tfrecord(self, example):
        # Note how we are defining the example structure here
        feature_description = {
            'log_mel': tf.io.FixedLenFeature([], tf.string),
            'ytid': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        # audio, sr = self.decode_audio(parsed_example['audio'])
        log_mel = tf.io.decode_raw(example['log_mel'], tf.float32)
        log_mel = tf.reshape(log_mel, (128, 626, 1))
        
        ytid =  example['ytid']
        labels = tf.io.decode_raw(example['labels'], tf.float32)

        return {'images': log_mel, 'labels': labels}  
    
    def to_jax(self, sample):
        sample['images'] = jnp.array(sample['images'], dtype=self.input_dtype)
        sample['images'] = jnp.transpose(sample['images'], (0, 3, 1, 2))
        sample['labels'] = jnp.array(sample['labels'], dtype=self.label_dtype)
        return sample['images'], sample['labels']
    
    def min_max_normalize(self, sample, epsilon=1e-7):
        min_val = -80.0
        max_val = 0.0
        sample['images'] = (sample['images'] - min_val) / (max_val - min_val + epsilon)
        # sample['images'] = tf.expand_dims(sample['images'], axis=-1)
        return sample
    
    
    
    def load_dataset(self, filenames, ordered=False, shuffle_buffer_size=1, drop_remainder=False,\
                augment=True):
        AUTO = tf.data.experimental.AUTOTUNE
        # tf.data runtime will optimize this parameter
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO, compression_type = 'ZLIB')

        options = tf.data.Options()
        if not ordered:
            options.experimental_deterministic = False
            
        # Step 1: Read in the data, shuffle and batching
        dataset = dataset.with_options(options)\
                    .map(self.read_labeled_tfrecord,
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.shuffle(shuffle_buffer_size)
        
        if augment:
            dataset = dataset.map(crop_resize, num_parallel_calls=AUTO)

        dataset = dataset.batch(self.args['batch_size'] * self.args['device_count'], drop_remainder=drop_remainder)
        dataset = dataset.repeat()
        
        # We exemplify augmentation using RandAugment
        if augment:
            dataset = dataset.map(mix_up, num_parallel_calls=AUTO)
            dataset = dataset.map(random_linear_fader, num_parallel_calls=AUTO)

        dataset = dataset.map(self.min_max_normalize)
        
        dataset = dataset.prefetch(AUTO)

        return map(self.to_jax, tfds.as_numpy(dataset))
    
    def get_dataloader(self, train):
        if train:
            return self.load_dataset(self.train_filenames)
        else:
            return None
        

class DataIterator:
    def __init__(self, my_iter: Iterable, len_data: int):
        self.my_iter = my_iter
        self.index = 0
        self.len_my_iter = len_data

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.len_my_iter:
            result = next(self.my_iter)
            self.index += 1
            return result
        else:
            self.index = 0
            raise StopIteration
    
    def __len__(self):
        return self.len_my_iter
    
    