import keras_cv
import tensorflow as tf

mix_up = keras_cv.layers.MixUp()


def random_linear_fader(sample):
    lms = sample["images"]
    # print((2.0 * tf.random.uniform((2,))))
    head_tail = (2.0 * tf.random.uniform((2,))) - 1.0 # gain * U(-1., 1) for two ends
    T = lms.shape[-3]
    slope = tf.linspace(head_tail[0], head_tail[1], T)
    slope = tf.reshape(slope, (T, 1, 1))
    sample["images"] = lms + slope
    return sample


class MyRandomCropAndResize(keras_cv.layers.RandomCropAndResize):
    def __init__(self, virtual_crop_scale, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_crop_scale = virtual_crop_scale
        
    def augment_image(self, image, transformation, **kwargs):
        virtual_crop_scale = self.virtual_crop_scale
        virtual_crop_size = [int(s * c) for s, c in zip(image.shape[-3:-1], virtual_crop_scale)]
        image = tf.image.resize_with_pad(image, virtual_crop_size[0], virtual_crop_size[1])
        
        return self._crop_and_resize(image, transformation)
    
    
    
crop_resize = MyRandomCropAndResize(virtual_crop_scale = (1.0,1.2), 
                                    target_size = (128,192),  
                                    crop_area_factor=(0.2, 1.0), 
                                    aspect_ratio_factor=(128/192, 128/192),
                                    )

# crop_resize

# def cut_mix_and_mix_up(samples):
#     # samples = cut_mix(samples, training=True)
#     samples = mix_up(samples, training=True)
    
#     return samples