from __future__ import print_function
import os
import time
import logging
import argparse
import tensorflow as tf
import numpy as np
import gym
from gym import wrappers

from models.model import Model

def config_logging(log_file):
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def create_model(session, logger, optimizer, learning_rate, checkpoint_dir, restore):
    model = Model(logger, optimizer, learning_rate, checkpoint_dir)

    if restore:
        restored = model.load(session)
        if not restored:
            logger.info("Created model with fresh parameters")
            session.run(tf.global_variables_initializer())
    else:
        logger.info("Created model with fresh parameters")
        session.run(tf.global_variables_initializer())

    return model

def train(learning_rate=1e-3, 
             results_dir=None, 
             checkpoint_dir=None,
             # network arguments
             batch_size=10,
             restore=False,
             optimizer='adam'
             ):
    tf.reset_default_graph()
    
    with tf.Session() as session:    
        model = create_model(session, logger, optimizer, learning_rate, checkpoint_dir, restore)

        file_writer = tf.summary.FileWriter(results_dir, session.graph)

    # TODO: Train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='acan')
    parser.add_argument('--batch_size', '-b', type=int, default=10)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--restore', '-restore', action='store_true')
    args = parser.parse_args()

    checkpoint_dir = os.path.join(os.getcwd(), 'results')
    results_dir = os.path.join(os.getcwd(), 'results', args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    train(
        learning_rate=args.learning_rate,
        results_dir=results_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=args.batch_size,
        restore=args.restore
        )

if __name__ == "__main__":
    log_file = os.path.join(os.getcwd(), 'results', 'train_out.log')
    logger = config_logging(log_file)

    main()
