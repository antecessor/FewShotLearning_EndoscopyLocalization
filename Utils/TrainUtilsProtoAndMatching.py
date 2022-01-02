from tqdm import tqdm
from DeepLearning.prototypicalNN import Prototypical
from DeepLearning.MatchingNN import MatchingNetwork

import time
import numpy as np
import tensorflow as tf

tf.config.gpu.set_per_process_memory_growth(True)


class TrainEngineMatchingAndPrototypicalNN(object):
    """
    Engine that launches training per epochs and episodes.
    Contains hooks to perform certain actions when necessary.
    """

    def __init__(self):
        self.hooks = {name: lambda state: None
                      for name in ['on_start',
                                   'on_start_epoch',
                                   'on_end_epoch',
                                   'on_start_episode',
                                   'on_end_episode',
                                   'on_end']}

    def train(self, loss_func, train_loader, val_loader, epochs, n_episodes, **kwargs):
        # State of the training procedure
        state = {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'loss_func': loss_func,
            'sample': None,
            'epoch': 1,
            'total_episode': 1,
            'epochs': epochs,
            'n_episodes': n_episodes,
            'best_val_loss': np.inf,
            'early_stopping_triggered': False
        }

        self.hooks['on_start'](state)
        for epoch in range(state['epochs']):
            self.hooks['on_start_epoch'](state)
            for _ in tqdm(range(state['n_episodes'])):
                x_support, y_support, x_query, y_query = train_loader.get_next_episode()
                state['sample'] = (x_support, y_support, x_query, y_query)
                self.hooks['on_start_episode'](state)
                self.hooks['on_end_episode'](state)
                state['total_episode'] += 1

            self.hooks['on_end_epoch'](state)
            state['epoch'] += 1

            # Early stopping
            if state['early_stopping_triggered']:
                print("Early stopping triggered!")
                break

        self.hooks['on_end'](state)
        print("Training succeed!")


class DataLoader(object):
    def __init__(self, data, n_classes, n_way, n_support, n_query):
        self.data = data
        self.n_way = n_way
        self.n_classes = n_classes
        self.n_support = n_support
        self.n_query = n_query

    def get_next_episode(self):
        n_examples = 20
        support = np.zeros([self.n_way, self.n_support, 28, 28, 1], dtype=np.float32)
        query = np.zeros([self.n_way, self.n_query, 28, 28, 1], dtype=np.float32)
        classes_ep = np.random.permutation(self.n_classes)[:self.n_way]

        for i, i_class in enumerate(classes_ep):
            selected = np.random.permutation(n_examples)[:self.n_support + self.n_query]
            support[i] = self.data[i_class, selected[:self.n_support]]
            query[i] = self.data[i_class, selected[self.n_support:]]

        return support, query


def train(X_train, y_train, modelType="proto", backBone=None):
    w, h, c = 28, 28, 1  # we can change to check different size input
    data = np.zeros([len(y_train), len(X_train), 28, 28, 1])
    for index, i_class in enumerate(y_train):
        for i_img, img in enumerate(X_train):
            data[i_class, i_img, :, :, :] = img

    data_loader = DataLoader(data,
                             n_classes=len(y_train),
                             n_way=len(y_train),
                             n_support=3,
                             n_query=5)

    if modelType == "proto":
        model = Prototypical(3, 3, w, h, c, ecnoder=backBone)
    else:
        model = MatchingNetwork(3, w, h, c, encoder=backBone)
    optimizer = tf.keras.optimizers.Adam(0.001)

    # Metrics to gather
    train_loss = tf.metrics.Mean(name='train_loss')
    val_loss = tf.metrics.Mean(name='val_loss')
    train_acc = tf.metrics.Mean(name='train_accuracy')
    val_acc = tf.metrics.Mean(name='val_accuracy')
    val_losses = []

    @tf.function
    def loss(support, query):
        loss, acc = model(support, query)
        return loss, acc

    @tf.function
    def train_step(loss_func, support, query):
        # Forward & update gradients
        with tf.GradientTape() as tape:
            loss, acc = model(support, query)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        # Log loss and accuracy for step
        train_loss(loss)
        train_acc(acc)

    @tf.function
    def val_step(loss_func, support, query):
        loss, acc = loss_func(support, query)
        val_loss(loss)
        val_acc(acc)

    # Create empty training engine
    train_engine = TrainEngineMatchingAndPrototypicalNN()

    # Set hooks on training engine
    def on_start(state):
        print("Training started.")

    train_engine.hooks['on_start'] = on_start

    def on_end(state):
        print("Training ended.")

    train_engine.hooks['on_end'] = on_end

    def on_start_epoch(state):
        print(f"Epoch {state['epoch']} started.")
        train_loss.reset_states()
        val_loss.reset_states()
        train_acc.reset_states()
        val_acc.reset_states()

    train_engine.hooks['on_start_epoch'] = on_start_epoch

    def on_end_epoch(state):
        print(f"Epoch {state['epoch']} ended.")
        epoch = state['epoch']
        template = 'Epoch {}, Loss: {}, Accuracy: {}, ' \
                   'Val Loss: {}, Val Accuracy: {}'
        print(
            template.format(epoch + 1, train_loss.result(), train_acc.result() * 100,
                            val_loss.result(),
                            val_acc.result() * 100))

        cur_loss = val_loss.result().numpy()
        if cur_loss < state['best_val_loss']:
            print("Saving new best model with loss: ", cur_loss)
            state['best_val_loss'] = cur_loss
            model.save("PositionModel_prototypical.h5")
        val_losses.append(cur_loss)

        # Early stopping
        patience = 100
        if len(val_losses) > patience \
                and max(val_losses[-patience:]) == val_losses[-1]:
            state['early_stopping_triggered'] = True

    train_engine.hooks['on_end_epoch'] = on_end_epoch

    def on_start_episode(state):
        if state['total_episode'] % 20 == 0:
            print(f"Episode {state['total_episode']}")
        support, query = state['sample']
        loss_func = state['loss_func']
        train_step(loss_func, support, query)

    train_engine.hooks['on_start_episode'] = on_start_episode

    def on_end_episode(state):
        # Validation
        val_loader = state['val_loader']
        loss_func = state['loss_func']
        for i_episode in range(1000):
            support, query = val_loader.get_next_episode()
            val_step(loss_func, support, query)

    train_engine.hooks['on_end_episode'] = on_end_episode

    time_start = time.time()

    train_engine.train(
        loss_func=loss,
        train_loader=data_loader,
        val_loader=data_loader,
        epochs=200,
        n_episodes=1000)
    time_end = time.time()

    elapsed = time_end - time_start
    h, min = elapsed // 3600, elapsed % 3600 // 60
    sec = elapsed - min * 60
    print(f"Training took: {h} h {min} min {sec} sec")
