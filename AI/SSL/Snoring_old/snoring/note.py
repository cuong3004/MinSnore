import os

# Thiết lập biến môi trường
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"


import jax

n_devices = jax.local_device_count() 

print(n_devices)