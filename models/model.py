import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import itertools

class Model:
    def __init__(self, logger, optimizer, learning_rate, checkpoint_dir):
        self.logger = logger

        self.init_global_step()
        self.build_model()

        self.x = tf.placeholder(shape=[None], name="x", dtype=tf.float32)
        self.y = tf.placeholder(shape=[None], name="y", dtype=tf.int32)

        self.loss = self.get_loss()
        self.optimizer = self.get_optimizer(optimizer, learning_rate)

        self.training_scalar = tf.summary.scalar("training_loss", self.loss)
        self.validation_scalar = tf.summary.scalar("validation_loss", self.loss)
        self.histogram_merged = tf.summary.merge_all()

        self.checkpoint_dir = checkpoint_dir
        self.saver = tf.train.Saver(var_list=tf.global_variables())

    def save(self, sess):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.saver.save(sess, self.checkpoint_dir + '/model', global_step=self.global_step_tensor)
        self.logger.info("Model saved")

    def load(self, session):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.logger.info("Loading model checkpoint {} ...\n".format(latest_checkpoint))
            self.saver.restore(session, latest_checkpoint)
            return True
        else:
            self.logger.info("Checkpoint not found")
            return False

    def init_global_step(self):
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')

    def build_mlp(self, input_placeholder, output_size, scope, num_layers=2, layer_size=64, 
            activation=tf.tanh, output_activation=None):
        with tf.variable_scope(scope):
            dense = input_placeholder
            for _ in range(num_layers):
                dense = tf.layers.dense(inputs=dense, units=layer_size, activation=activation)

            return tf.layers.dense(inputs=dense, units=output_size, activation=output_activation)

    def build_model(self):
        # TODO: output_size
        logits_na = self.build_mlp(input_placeholder=self.x, output_size=1,
                scope="discrete_policy_network",
                activation=tf.nn.relu)
        sampled_ac = tf.squeeze(tf.multinomial(logits_na, 1), axis=[1])
        logprob_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=logits_na)

        return logprob_n, sampled_ac

    def get_optimizer(self, optimizer, learning_rate):
        self.logger.info("Using %s optimizer" % optimizer)
        if optimizer == "adam":
            return tf.train.AdamOptimizer(learning_rate).minimize(self.loss,
                global_step=self.global_step_tensor)
        elif optimizer == "adagrad":
            return tf.train.AdagradOptimizer(learning_rate).minimize(self.loss,
                global_step=self.global_step_tensor)
        elif optimizer == "rmsprop":
            return tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss,
                global_step=self.global_step_tensor)
        else:
            return tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss,
                global_step=self.global_step_tensor)

    def get_loss(self):
        # TODO: calculate loss
        loss = tf.reduce_mean(tf.multiply(1, 1))
        return loss

    def validate(self, sess, batch_x, batch_y):
        return sess.run([self.loss, self.validation_scalar],
                        feed_dict={self.x:batch_x,
                                    self.y: batch_y})

    def predict(self, sess, batch_x):
        return sess.run(self.y,
                        feed_dict={self.x:batch_x})

    def update(self, sess, batch_x, batch_y, advantages, keep_prob):
        loss, training_scalar, _, histogram_merged, _ = sess.run([self.loss, self.training_scalar, self.histogram_merged, self.optimizer],
                        feed_dict={self.x: batch_x,
                                    self.y: batch_y})
        return loss, training_scalar, histogram_merged

    def test_run(self, sess, env, max_steps):
        obvs = []
        actions = []
        reward = 0.

        obv = env.reset()
        for steps in itertools.count() :
            obvs.append(obv)
            actions.append(self.predict(sess, np.expand_dims(obv,axis=0))[0])
            obv, r, done, _ = env.step(actions[-1])
            reward += r
            if steps >= max_steps or done:
                break

        experience = {'observations': np.stack(obvs,axis=0),
                      'actions': np.squeeze(np.stack(actions,axis=0)),
                      'reward':reward}
        return experience