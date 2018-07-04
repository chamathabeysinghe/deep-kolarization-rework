from keras import backend as K
from keras.engine import Layer


class FusionLayer(Layer):
    def __init__(self, **kwargs):
        super(FusionLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        enc_out, incep_out = inputs
        enc_out_shape = K.shape(enc_out)
        batch_size, h, w = [enc_out_shape[i] for i in range(3)]

        def _repeat(emb):
            # keep batch_size axis unchanged
            # while replicating features h*w times
            emb_rep = K.tile(emb, [1, h * w])
            return K.reshape(emb_rep, (batch_size, h, w, emb.shape[1]))

        incep_rep = _repeat(incep_out)
        return K.concatenate([enc_out, incep_rep], axis=3)

    def compute_output_shape(self, input_shapes):
        enc_out_shape, incep_out_shape = input_shapes

        # Must have 3 tensors as input
        assert input_shapes and len(input_shapes) == 2

        # Batch size of the two tensors must match
        assert enc_out_shape[0] == incep_out_shape[0]

        # batch_size, time_steps, h, w, enc_out_depth = map(lambda x: -1 if x == None else x, enc_out_shape)
        batch_size, h, w, enc_out_depth = enc_out_shape
        final_depth = enc_out_depth + incep_out_shape[1]
        return batch_size, h, w, final_depth
