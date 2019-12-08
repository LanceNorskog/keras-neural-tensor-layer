import tensorflow.keras.layers
import tensorflow.keras.backend as K
import tensorflow_hub as hub

class USE(Lambda):
    """Lambda Wrapper for Universal Sentence Embedding.
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
        use_url: String url for Universal Sentence Embedding, default to large-3
    # Input shape
        1D tensor with shape: (batch_size, 1) of type `tf.string`
    # Output shape
        1D tensor with shape: `(batch_size, 512)`.
    """
    embed = None

    def UniversalEmbedding(x):
        return embed(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

    def __init__(self, 
                use_url="https://tfhub.dev/google/universal-sentence-encoder-large/3",
                 **kwargs):
        self.use_url = use_url
        self.embed = hub.Module(module_url)
        embed_size = embed.get_output_info_dict()['default'].get_shape()[1].value
        assert embed_size == 512
        super(USE, self).__init__(UniversalEmbedding, output_dim=(512, ), **kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert input_shape[-1] == 1
        # assert type is string?
        super(USE, self).build(input_shape)

    def get_config(self):
        config = {
            'use_url': self.use_url
        }
        base_config = super(USE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

