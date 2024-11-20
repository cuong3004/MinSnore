import os
import warnings
import tensorflow as tf
import jax 
from jax import numpy as jnp
from jax.lib import xla_bridge
platform = xla_bridge.get_backend().platform
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

import requests 
from jax.config import config

# print(config.FLAGS.jax_xla_backend)
# print(config.FLAGS.jax_backend_target)
# config.FLAGS.jax_xla_backend = "tpu_driver"
# config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
if platform == "tpu":
    # config.update('jax_default_matmul_precision', 'bfloat16')
    use_tpu = False
else:
    use_tpu = False
# print(os.environ['TPU_NAME'])
# print('Registered (Kaggle) TPU:', config.FLAGS.jax_backend_target)

# use_tpu = True
# platform
# Seems like the new one will break jax
# if 'TPU_NAME' in os.environ and 'KAGGLE_DATA_PROXY_TOKEN' in os.environ:
#     use_tpu = True
    
#     import requests 
#     from jax.config import config
#     if 'TPU_DRIVER_MODE' not in globals():
#         url = 'http:' + os.environ['TPU_NAME'].split(':')[1] + ':8475/requestversion/tpu_driver_nightly'
#         resp = requests.post(url)
#         TPU_DRIVER_MODE = 1
#     config.FLAGS.jax_xla_backend = "tpu_driver"
#     config.FLAGS.jax_backend_target = os.environ['TPU_NAME']
#     # Enforce bfloat16 multiplication
#     config.update('jax_default_matmul_precision', 'bfloat16')
#     print('Registered (Kaggle) TPU:', config.FLAGS.jax_backend_target)
# else:
#     use_tpu = False
# /////////////////////////////////////////

args = {
    'experiment_name': 'starter',
    # Efficientnetv2-m
    'model': 'm',
    'batch_size': 128,# if use_tpu else 8, 
    'epochs': 16,
    'base_lr': 7e-5, # should directly correspond to `batch_size`
    
    #use_tpu
    "use_tpu": use_tpu,
    "input_dtype": jnp.bfloat16 if use_tpu else jnp.float32,
    # Data / augmentation 
    'img_size': 512, # 192, 224, 331, 512
    # The actual label number is 104
    'num_labels': 104 if not use_tpu else 128,
    'saving_dir': '/kaggle/working/',
    'device_count' : jax.device_count(),
    
    # Debugging-purposes
    'sanity_check': False,
    
    #dataset
    'train_dirs': [
        # 'gs://kds-6ac4b92104b0b81a132d18c52e1438047c5fee2726f035421b1e6071',
        'gs://kds-22d045b0ae3553d8b0bc903999aa8ce544593e74ecc99fe9149f04a9',
        'gs://kds-6f80ca8eaa77391ec3dcff20e7842873b98ddc8abb044d5dd0e99f23',
        'gs://kds-52dad802a449570726b10367ad8bd3bd62fe04e5c671e6be21ed2494',
        'gs://kds-3b07f926da9ce362dee51e808ab01cf8209bafff6c5b37c3eb21b47f',
        'gs://kds-f0643c7ee21cc8c219275deae92d538f15b0777b0e5acae5ac712593',
        'gs://kds-38363588642f6610c99259e1faba665d5e569457875a8cbe34445755',
        'gs://kds-e5875d1c6264c5483800c218b0199ca946e019b9e9e6cb141f468594'
    ]
}

# The effective lr should linearly scale with batch size
args['lr'] = args['base_lr'] * args['device_count']
if args['sanity_check']:
    args['epochs'] = 1

# Data-specific, should change for each dataset 
args['data_dir'] = '/kaggle/input/tpu-getting-started/tfrecords-jpeg-'\
                    + str(args['img_size'])+'x'+str(args['img_size'])
args['train_dir'] = os.path.join(args['data_dir'], 'train')
args['val_dir'] = os.path.join(args['data_dir'], 'val')
args['test_dir'] = os.path.join(args['data_dir'], 'test')

print('Running on', args['device_count'], 'processors')
print(jax.devices())

print(args)
