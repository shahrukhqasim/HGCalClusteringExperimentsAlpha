import tensorflow as tf
from LayersRagged import RaggedGravNet, RaggedConstructTensor


class GravnetModel(tf.keras.models.Model):
    def __init__(self, clustering_space_dimensions=2):
        super(GravnetModel, self).__init__()
        self.glayer_1 = RaggedGravNet()
        self.glayer_2 = RaggedGravNet()
        self.glayer_3 = RaggedGravNet()
        self.glayer_4 = RaggedGravNet()
        self.glayer_5 = RaggedGravNet()

        self.ragged_constructor = RaggedConstructTensor()

        self.output_dense_clustering = tf.keras.layers.Dense(80, activation=tf.nn.relu)
        self.output_dense_beta = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        # self.output_dense_3 = tf.keras.layers.Dense(num_outputs)

    def call(self, inputs, rowsplits):


        inputs, rowsplits = self.ragged_constructor((inputs, rowsplits))

        rowsplits = tf.cast(rowsplits, tf.int32)

        x = self.glayer_1((inputs, rowsplits))
        x = self.glayer_2((x, rowsplits))
        x = self.glayer_3((x, rowsplits))
        x = self.glayer_4((x, rowsplits))
        x = self.glayer_5((x, rowsplits))

        return self.output_dense_clustering(x), tf.squeeze(self.output_dense_beta(x), axis=-1)

