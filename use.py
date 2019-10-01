import tensorflow.keras.layers.Layer
import tensorflow_hub as hub


class USE(Layer):
    """Wrapper for Universal Sentence Embedding.
    # Example
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    # Arguments
        use_url: link for desired USE version. Defaults to Large-3.
    # Input shape
        1D tensor with shape: (batch_size, 1) of type `tf.string`
    # Output shape
        1D tensor with shape: `(batch_size, 512)`.
    """

    def __init__(self, 
                use_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                 **kwargs):
        super(USE, self).__init__(**kwargs)
        self.use_url = use_url

    def build(self, input_shape):
        # Mysterious tf-hub stuff
        self.embed = hub.Module(module_url)
        embed_size = self.embed.get_output_info_dict()['default'].get_shape()[1].value
        print('USE embed size: {}'.format(embed_size))
        assert len(input_shape) == 2
        assert input_shape[-1] == 1

        self.lambda = self.add_weight(shape=(input_dim, self.units), initializer=self.kernel_initializer, name='kernel', regularizer=self.kernel_regularizer, constraint=self.kernel_constraint)
        # ??
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def UniversalEmbedding(x):
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

    def call(self, inputs):
        assert len(inputs) == 1
        return UniversalEmbedding(inputs[0])

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'use_url': self.use_url
        }
        base_config = super(Layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

