
# coding: utf-8

# In[30]:

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import collections
#import logging
#from tensorflow.models.rnn import rnn
import numpy as np
import tensorflow as tf
import os

from tensorflow.models.rnn.ptb import reader
import tensorflow.models.rnn 
flags = tf.flags
logging = tf.logging
#logger.propagate = False
#Configurations

init_scale = 0.1
learning_rate = 0.85
max_grad_norm = 5
num_layers = 4
num_steps = 3 #windows size
hidden_size = 200
max_epoch = 4
max_max_epoch = 10
keep_prob = 0.9
lr_decay = 0.5
batch_size = 1
vocab_size = 12000

#output of prediction and logging
output_file = open("predictions.txt", "w")
output_log = open("output_log.txt","w")


# In[31]:

class PTBModel(object):
    """The PTB model."""

    def __init__(self, is_training):
        size = hidden_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
        if is_training and keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * num_layers)

        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/gpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size])
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob)

        # Simplified version of tensorflow.models.rnn.rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # from tensorflow.models.rnn import rnn
        # inputs = [tf.squeeze(input_, [1])
        #           for input_ in tf.split(1, num_steps, inputs)]
        # outputs, state = rnn.rnn(cell, inputs, initial_state=self._initial_state)
        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        
        
        
        loss = tf.nn.seq2seq.sequence_loss_by_example( [logits],
            [tf.reshape(self._targets, [-1])], [tf.ones([batch_size * num_steps])])
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        #Storing the probabilities and logits
        self.probabilities = probabilities =  tf.nn.softmax(logits)
        self.logits = logits

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op


# In[32]:

def run_epoch(session, m, data, eval_op, verbose=True,test=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // batch_size) - 1) // num_steps
    print("Epoch size: %d"%epoch_size)
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    correct_predictions = 0
    total_predictions = 0
    correct_predictions_4 = 0
    eos_ind = words.index("<eos>")
    for step, (x, y) in enumerate(reader.ptb_iterator(data,batch_size, num_steps)):
        if eos_ind in x[0]:
            continue
        if eos_ind in y[0]:
            continue
        cost, state,probs, logits, _ = session.run([m.cost, m.final_state, m.probabilities, m.logits, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
        costs += cost
        iters += num_steps
        
        
        #chosen_word = np.argmax(probs, 1)
        chosen_word = probs.argsort()[-4:][::-1]
        #print("Chosen word")
        #print(len(chosen_word[-1]))
        #print(chosen_word)
        #print(chosen_word[-1])
        
        out_lg = '[ '
        for i in x[0]:
            out_lg = out_lg + ' ' + words[i]
        out_lg = out_lg + ' ] [ '
        
        flag_in_range = True
        for i in range(4):
            if chosen_word[-1][i] > vocab:
                print("There's a problem here %d"%chosen_word[-1][i])
                flag_in_range = False
        if flag_in_range:
            out_lg = out_lg + words[chosen_word[-1][0]] + ' , '
            out_lg = out_lg + words[chosen_word[-1][1]] + ' , '
            out_lg = out_lg + words[chosen_word[-1][2]] + ' , '
            out_lg = out_lg + words[chosen_word[-1][3]] + ' ] '
        
        #out_lg = out_lg + str(probs[chosen_word[-1][0]]) + ' , '
        #out_lg = out_lg + str(probs[chosen_word[-1][1]]) + ' , '
        #out_lg = out_lg + str(probs[chosen_word[-1][2]]) + ' ]'
        
        output_log.write("%s %.3f\n"%(out_lg, cost))
        if test and flag_in_range:
            Fragmented_sentence = False
            if not Fragmented_sentence:
                total_predictions += 1
                
                print(step)
                inp = ''
                for w in x[0]:
                    inp = inp + ' ' +  words[w]
                print("Input: %s" % inp)

                out = ''
                for w in y[0]:
                    out = out +  ' ' + words[w]
                print("Output : %s" % out)
                
                out = ''
                for w in y[0]:
                    out = out + ' ' + words[w]
                out = out + ' ' + words[chosen_word[-1][0]]
                output_file.write(out + "\n")
                if y[0][-1] == chosen_word[-1][0]:
                    correct_predictions += 1
                if y[0][-1] in chosen_word[-1][0:4]:
                    correct_predictions_4 += 1
                print("Prediction: %s: \n" % words[chosen_word[- 1][0]])
            
            
        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %(step * 1.0 / epoch_size, np.exp(costs / iters),
                     iters * batch_size / (time.time() - start_time)))
            
            print("Probabilities shape: %s, Logits shape: %s" %  (probs.shape, logits.shape) )
            

            print("Batch size: %s, Num steps: %s" % (batch_size, num_steps))
    if test:
        perc = (correct_predictions + 0.0) / (0.0 + total_predictions)
        perc4 = (correct_predictions_4 + 0.0) / (0.0 + total_predictions)
        print("Total prediction: %d, Correct prediction: %d, percentage: %.3f"%(total_predictions, correct_predictions, perc))
        print("Top4: Total prediction: %d, Correct prediction: %d, percentage: %.3f"%(total_predictions, correct_predictions_4, perc_4))
    return np.exp(costs / iters)


# In[33]:

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

data_path = 'simple-examples/dataprod1/'

train_path = os.path.join(data_path, "ptb.train.txt")
data = _read_words(train_path)

counter = collections.Counter(data)
count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

words, _ = list(zip(*count_pairs))


raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, vocab = raw_data

print(vocab)


for i in range(25):
    print("[ %d , %d ,  %s ]" % (i,test_data[i], words[test_data[i]]))

print(len(test_data))


# In[ ]:




# In[34]:

with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = PTBModel(is_training=True)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = PTBModel(is_training=False)
        mtest = PTBModel(is_training=False)

    tf.initialize_all_variables().run()
    writer = tf.train.SummaryWriter("output_graph", session.graph)
    for i in range(max_max_epoch):
        lr_decay = lr_decay ** max(i - max_epoch, 0.0)
        m.assign_lr(session, learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                   verbose=True, test = False)
        print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
        valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op(),test = False)
        print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
    saver = tf.train.Saver()
    #Global step = max_max_epoch + 1
    saver.save(session, 'word_predict.ckpt')

    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op(), test = True)
    print("Test Perplexity: %.3f" % test_perplexity)
    output_file.close()


# In[13]:

a = np.array([2,2,7,4,5,6])


# In[29]:

len(words)


# In[ ]:



