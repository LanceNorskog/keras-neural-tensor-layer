from keras.layers import Layer, Input
import tensorflow as tf
import tensorflow_hub as hub

embed = None
class USE(Layer):
    def __init__(self, 
                use_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                 **kwargs):
        self.use_url = use_url
        self.strtype = "string"
        super(USE, self).__init__(**kwargs)

    def build(self, input_shape):
        global embed
        # Mysterious tf-hub stuff
        if embed == None:
            embed = hub.Module(self.use_url)
        self.embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value
        print('USE embed size: {}'.format(self.embed_size))
        super(USE, self).build(input_shape)

    def call(self, inputs):
        global embed
        #assert len(inputs.shape) == 1
        return embed(tf.squeeze(tf.cast(inputs, self.strtype)), signature="default", as_dict=True)["default"]

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.embed_size
        return tuple(output_shape)

    def get_config(self):
        config = {
            'use_url': self.use_url
        }
        return config

    def get_input(self):
        return Input(shape=(1,), dtype=self.strtype)

