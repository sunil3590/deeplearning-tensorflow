#!/usr/bin/python

""" Auto Encoder Example
Auto encode words
"""
import nltk
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class AutoEncoder:
    def __init__(self):
        # Get input and other data structures
        self.text, self.word_to_index, self.index_to_word = AutoEncoder._collect_fake_input_data()

        # Network Parameters
        self.n_input, self.n_hidden_1, self.weights, self.biases = self._configure_network()

        # Training parameters
        self.learning_rate, self.training_epochs, self.batch_size = AutoEncoder._configure_trainer()

        # Tensorflow session
        self.sess = tf.Session()

        # TF placehoders, variables and network
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_input])
        self.emb = self._encoder(self.x)
        self.y_ = self._decoder(self.emb)

        # Launch the graph
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def _collect_input_data():
        # Import nltk data
        text = nltk.corpus.gutenberg.words('austen-emma.txt')
        return AutoEncoder._process_text_data(text)

    @staticmethod
    def _collect_mini_input_data():
        text = nltk.corpus.gutenberg.words('austen-emma.txt')
        text = text[0:1000]
        return AutoEncoder._process_text_data(text)

    @staticmethod
    def _collect_fake_input_data():
        text = "One of the most satisfying observations about the CNN encoder is that the model reconstructs well " \
               "formed English sentences significantly better than it does garbled sentences which is evidence that " \
               "the CNN features encode useful syntactic structures nonexistent in random input for unfolding the " \
               "hidden representation "
        text = text.split(" ")
        return AutoEncoder._process_text_data(text)

    @staticmethod
    def _process_text_data(text):
        words = set(text)
        word_to_index = dict()
        index_to_word = dict()
        for word in words:
            index = len(word_to_index)
            word_to_index[word] = index
            index_to_word[index] = word
        return text, word_to_index, index_to_word

    def _configure_network(self):
        n_input = len(self.word_to_index)
        n_hidden_1 = n_input / 4

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b1': tf.Variable(tf.random_normal([n_input])),
        }

        print "# of nodes in encoder layers :", n_input, n_hidden_1

        return n_input, n_hidden_1, weights, biases

    @staticmethod
    def _configure_trainer():
        learning_rate = 2
        training_epochs = 100
        batch_size = 1

        print "Training parameters : Learning rate -", learning_rate, " Training epochs -", training_epochs, \
            " Batch size -", batch_size

        return learning_rate, training_epochs, batch_size

    def _get_one_hot(self, words):
        one_hots = list()
        for word in words:
            one_hot = list()
            for i in range(0, len(self.word_to_index)):
                one_hot.append(0)
            one_hot[self.word_to_index[word]] = 1
            one_hots.append(one_hot)
        return one_hots

    def _get_word(self, one_hot):
        for index in range(0, len(one_hot)):
            if one_hot[index] == 1:
                return self.index_to_word[index]

    # Building the encoder
    def _encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        return layer_1

    # Building the decoder
    def _decoder(self, x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        return layer_1

    def train(self):
        # Define loss and optimizer, minimize the squared error
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.y_, labels=self.y))
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cross_entropy)

        total_batch = len(self.text) / self.batch_size - 1
        # Training cycle
        for epoch in range(self.training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                start = i * self.batch_size
                end = start + self.batch_size
                batch_xs = self._get_one_hot(self.text[start:end])
                batch_ys = batch_xs
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = self.sess.run([optimizer, cross_entropy], feed_dict={self.x: batch_xs, self.y: batch_ys})
            # Display logs
            if epoch % 5 == 0:
                print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.5f}".format(c)

    def evaluate(self):
        batch_xs = self._get_one_hot(self.word_to_index.keys())
        preds = self.y_.eval(feed_dict={self.x: batch_xs}, session=self.sess)
        total = 0
        correct = 0
        for pred, exp in zip(preds, batch_xs):
            pred_word = self.index_to_word[self.sess.run(tf.nn.top_k(pred, 1)).indices[0]]
            exp_word = self.index_to_word[self.sess.run(tf.nn.top_k(exp, 1)).indices[0]]
            total += 1
            if exp_word == pred_word:
                correct += 1
            else:
                print exp_word, "->", pred_word
        print "Accuracy = ", float(correct)/total

    def plot_words(self, fname):
        words = self.word_to_index.keys()
        batch_xs = self._get_one_hot(words)
        embs = self.emb.eval(feed_dict={self.x: batch_xs}, session=self.sess)

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        low_dim_embs = tsne.fit_transform(embs)
        plt.figure(figsize=(12, 12))  # in inches
        for i, label in enumerate(words):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.savefig(fname)


def main():
    autoenc = AutoEncoder()
    autoenc.plot_words("autoenc_pre_embs_tsne.png")
    autoenc.train()
    autoenc.evaluate()
    autoenc.plot_words("autoenc_post_embs_tsne.png")

if __name__ == "__main__":
    main()
