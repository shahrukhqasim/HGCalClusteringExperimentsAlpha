# -*- coding: utf-8 -*-

import tensorflow as tf
from LayersRagged import RaggedGravNet, RaggedConstructTensor
import DeepJetCore.DataCollection as dc
from model_gravnet_alpha import GravnetModel
import os
from ops import evaluate_loss
import uuid
import time





my_model = GravnetModel()
print("Model object made now initializing the parameters etc with one call")

print("But... to do that, we first need to load data. So loading data. Gonna take a while.")


data = dc.DataCollection('/eos/cms/store/cmst3/group/hgcal/CMG_studies/pepr/50_part_sample/testdata/dataCollection.djcdc')
data.setBatchSize(40000)
data.invokeGenerator()
nbatches = data.generator.getNBatches()
print("The data has",nbatches,"batches.")
gen = data.generatorFunction()

batch = gen.next()
print("First call happening now")
my_model.call(batch[0][0], batch[0][1])
print("Done... all good now I guess")


print(batch[1][0][0:-1][:,0])
print(batch[1][0][0:-1][:,0])


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

ragged_constructor = RaggedConstructTensor()


training_number = uuid.uuid4().hex
command = 'mkdir -p summaries_def/%s' % training_number
os.system(command)


writer = tf.summary.create_file_writer("summaries_def/%s"%training_number)


for iteration in range(1000000000):
    # batch = gen.next()
    print("Iteration ",iteration+1)

    with tf.GradientTape() as tape:
        start = time.time()
        clustering_space, beta_values = my_model.call(batch[0][0], batch[0][1])
        end = time.time()
        print("\tModel execution took", (end - start) * 1000, "milli seconds")

        row_splits = batch[0][1][:,0]
        classes, row_splits = ragged_constructor((batch[1][0][:, 0][..., tf.newaxis], row_splits))
        classes = classes[:, 0]
        row_splits = tf.convert_to_tensor(row_splits)
        row_splits = tf.cast(row_splits, tf.int32)

        start = time.time()
        loss, losses = evaluate_loss(clustering_space, beta_values, classes, row_splits, Q_MIN=0.005, S_B=0.3)
        end = time.time()
        print("\tLoss evaluation took", (end - start) * 1000, "milli seconds")

        beta_loss_first_term, beta_loss_second_term, attractive_loss, repulsive_loss = losses
    grads = tape.gradient(loss, my_model.trainable_variables)
    start = time.time()
    optimizer.apply_gradients(zip(grads, my_model.trainable_variables))
    end = time.time()
    print("\tBackpropagation took", (end - start) * 1000, "milli seconds")

    tf.summary.scalar("loss", loss, step=iteration)
    tf.summary.scalar("beta loss premier trimestre", loss, step=iteration)
    tf.summary.scalar("beta loss duxieme trimestre", loss, step=iteration)
    tf.summary.scalar("répulsive loss", loss, step=iteration)
    tf.summary.scalar("attrayante loss", loss, step=iteration)

    if iteration % 1000 == 0:
        my_model.save_weights('checkpoints/%s/checkpoint' % training_number)
        with open('checkpoints/%s/checkpoint_numero_d_iteration.txt'%training_number, 'w') as f:
            f.write('%d' % iteration)

input()