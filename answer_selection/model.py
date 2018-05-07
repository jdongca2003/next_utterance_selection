# Copyright (c) 2017 AT&T Intellectual Property. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS

def get_embeddings(vocab):
    print("get_embedding")
    initializer = load_word_embeddings(vocab, FLAGS.embedding_dim)
    return tf.constant(initializer, name="word_embedding")

def get_char_embedding(charVocab):
    print("get_char_embedding")
    char_size = len(charVocab)
    embeddings = np.zeros((char_size, char_size), dtype='float32')
    for i in range(1, char_size):
        embeddings[i, i] = 1.0

    return tf.constant(embeddings, name="word_char_embedding")

def load_embed_vectors(fname, dim):
    vectors = {}
    for line in open(fname, 'rt'):
        items = line.strip().split(' ')
        if len(items[0]) <= 0:
            continue
        vec = [float(items[i]) for i in range(1, dim+1)]
        vectors[items[0]] = vec

    return vectors

def load_word_embeddings(vocab, dim):
    vectors = load_embed_vectors(FLAGS.embeded_vector_file, dim)
    vocab_size = len(vocab)
    embeddings = np.zeros((vocab_size, dim), dtype='float32')
    for word, code in vocab.items():
        if word in vectors:
            embeddings[code] = vectors[word]
        #else:
        #    embeddings[code] = np.random.uniform(-0.25, 0.25, dim) 

    return embeddings 


def lstm_layer(inputs, input_seq_len, rnn_size, dropout_keep_prob, scope, scope_reuse=False):
    with tf.variable_scope(scope, reuse=scope_reuse) as vs:
        fw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        fw_cell  = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
        bw_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0, state_is_tuple=True, reuse=scope_reuse)
        bw_cell  = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout_keep_prob)
        rnn_outputs, rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                                                inputs=inputs,
                                                                sequence_length=input_seq_len,
                                                                dtype=tf.float32)
        return rnn_outputs, rnn_states


def question_answer_similarity_matrix(question, answer):
    q_len = question.get_shape()[1].value
    a_len = answer.get_shape()[1].value
    dim = question.get_shape()[2].value

    q_w = question

    #answer : batch_size * a_len * dim
    #[batch_size, dim, q_len]
    q2 = tf.transpose(q_w, perm=[0,2,1])

    #[batch_size, a_len, q_len]
    similarity = tf.matmul(answer, q2, name='similarity_matrix')

    return similarity


def self_attended(similarity_matrix, inputs):
    #similarity_matrix: [batch_size, len, len]
    #inputs: [batch_size, len, dim]

    attended_w = tf.nn.softmax(similarity_matrix, dim=-1)

    #[batch_size, len, dim]
    attended_out = tf.matmul(attended_w, inputs)
    return attended_out

def attended_answers(similarity_matrix, questions):
    #similarity_matrix: [batch_size, a_len, q_len]
    #questions: [batch_size, q_len, dim]

    #[batch_size, a_len, q_len]
    attention_weight_for_q = tf.nn.softmax(similarity_matrix, dim=-1)

    #[batch_size, a_len, dim]
    attended_answers = tf.matmul(attention_weight_for_q, questions)
    return attended_answers

def attended_questions(similarity_matrix, answers):
    #similarity_matrix: [batch_size, a_len, q_len]
    #answers: [batch_size, a_len, dim]

    #[batch_size, q_len, a_len]
    attention_weight_for_a = tf.nn.softmax(tf.transpose(similarity_matrix, perm=[0,2,1]), dim=-1)

    #[batch_size, q_len, dim]
    attended_questions = tf.matmul(attention_weight_for_a, answers)
    return attended_questions


class ESIM(object):
    def __init__(
      self, sequence_length, vocab_size, embedding_size, vocab, rnn_size, maxWordLength, charVocab, l2_reg_lambda=0.0):

        #question
        self.question = tf.placeholder(tf.int32, [None, sequence_length], name="question")
        #answer
        self.answer = tf.placeholder(tf.int32, [None, sequence_length], name="answer")

        self.target = tf.placeholder(tf.float32, [None], name="target")

        self.target_loss_weight = tf.placeholder(tf.float32, [None], name="target_weight")

        self.question_len = tf.placeholder(tf.int32, [None], name="question_len")
        self.answer_len = tf.placeholder(tf.int32, [None], name="answer_len")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.extra_feature = tf.placeholder(tf.float32, [None, 2], name="extra_feature")

        self.q_word_feature = tf.placeholder(tf.float32, [None, sequence_length, 2], name="question_word_feature")
        self.a_word_feature = tf.placeholder(tf.float32, [None, sequence_length, 2], name="answer_word_feature")

        self.q_charVec = tf.placeholder(tf.int32, [None, sequence_length, maxWordLength], name="question_char")
        self.q_charLen = tf.placeholder(tf.int32, [None, sequence_length], name="question_char_len")

        self.a_charVec = tf.placeholder(tf.int32, [None, sequence_length, maxWordLength], name="answer_char")
        self.a_charLen =  tf.placeholder(tf.int32, [None, sequence_length], name="answer_char_len")

        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = get_embeddings(vocab) 
            question_embedded = tf.nn.embedding_lookup(W, self.question)
            answer_embedded = tf.nn.embedding_lookup(W, self.answer)

        with tf.device('/cpu:0'), tf.name_scope('char_embedding'):
            char_W = get_char_embedding(charVocab)
            #[batch_size, q_len, maxWordLength, char_dim]
            question_char_embedded = tf.nn.embedding_lookup(char_W, self.q_charVec)

            #[batch_size, a_len, maxWordLength, char_dim]
            answer_char_embedded   = tf.nn.embedding_lookup(char_W, self.a_charVec)


        charRNN_size=40
        charRNN_name="char_RNN"
        char_dim = question_char_embedded.get_shape()[3].value
        question_char_embedded = tf.reshape(question_char_embedded, [-1, maxWordLength, char_dim])
        question_char_len      = tf.reshape(self.q_charLen, [-1])
        answer_char_embedded  = tf.reshape(answer_char_embedded, [-1, maxWordLength, char_dim])
        answer_char_len       = tf.reshape(self.a_charLen, [-1])

        char_rnn_output1, char_rnn_states1 = lstm_layer(question_char_embedded, question_char_len, charRNN_size, self.dropout_keep_prob, charRNN_name, scope_reuse=False)
        char_rnn_output2, char_rnn_states2 = lstm_layer(answer_char_embedded, answer_char_len, charRNN_size, self.dropout_keep_prob, charRNN_name, scope_reuse=True)

        question_char_state = tf.concat(axis=1, values=[char_rnn_states1[0].h, char_rnn_states1[1].h])
        char_embed_dim = 2 * charRNN_size
        question_char_state = tf.reshape(question_char_state, [-1, sequence_length, char_embed_dim])

        answer_char_state = tf.concat(axis=1, values=[char_rnn_states2[0].h, char_rnn_states2[1].h] )
        answer_char_state = tf.reshape(answer_char_state, [-1, sequence_length, char_embed_dim])

        rnn_scope_name = "bidirectional_rnn"

        question_embedded = tf.concat(axis=2, values=[question_embedded, question_char_state])
        answer_embedded   = tf.concat(axis=2, values=[answer_embedded, answer_char_state])

        print("shape of question_embedded");
        print(question_embedded.get_shape())

        rnn_output1, rnn_states1 = lstm_layer(question_embedded, self.question_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=False)
        rnn_output2, rnn_states2 = lstm_layer(answer_embedded, self.answer_len, rnn_size, self.dropout_keep_prob, rnn_scope_name, scope_reuse=True)

        #[batch_size, question_len, dim]
        question_output = tf.concat(axis=2, values=rnn_output1)

        #[batch_size, answer_len, dim]
        answer_output   = tf.concat(axis=2, values=rnn_output2)


        HOPS = 1

        for i in range(HOPS):
            #[batch_size, answer_len, question_len]
            similarity = question_answer_similarity_matrix(question_output, answer_output)

            #[batch_size, answer_len, dim]
            attended_answer_output = attended_answers(similarity, question_output)

            #[batch_size, question_len, dim]
            attended_question_output = attended_questions(similarity, answer_output)

            m_a = tf.concat(axis=2, values=[answer_output, attended_answer_output, tf.multiply(answer_output, attended_answer_output), answer_output-attended_answer_output])
            m_q = tf.concat(axis=2, values=[question_output, attended_question_output, tf.multiply(question_output, attended_question_output), question_output-attended_question_output])
            rnn_scope_layer2 = 'bidirectional_rnn_{}'.format(i+2)
            rnn_size_layer_2 = rnn_size
            rnn_output_q_2, rnn_states_q_2 = lstm_layer(m_q, self.question_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_layer2, scope_reuse=False)
            rnn_output_a_2, rnn_states_a_2 = lstm_layer(m_a, self.answer_len, rnn_size_layer_2, self.dropout_keep_prob, rnn_scope_layer2, scope_reuse=True)

            question_output = tf.concat(axis=2, values=rnn_output_q_2)
            answer_output   = tf.concat(axis=2, values=rnn_output_a_2)


        question_output_2 = tf.concat(axis=2, values=rnn_output_q_2)
        answer_output_2   = tf.concat(axis=2, values=rnn_output_a_2)

        final_question_max = tf.reduce_max(question_output_2, axis=1)
        final_answer_max   = tf.reduce_max(answer_output_2, axis=1)

        layer_q_last_state = tf.concat(axis=1, values=[rnn_states_q_2[0].h, rnn_states_q_2[1].h])
        layer_a_last_state = tf.concat(axis=1, values=[rnn_states_a_2[0].h, rnn_states_a_2[1].h])

        with tf.device('/gpu:0'), tf.name_scope("convolution-1"):

            joined_feature =  tf.concat(axis=1, values=[final_question_max, final_answer_max, layer_q_last_state, layer_a_last_state])
            print("shape of joined feature")
            print(joined_feature.get_shape())

            hidden_input_size = joined_feature.get_shape()[1].value

            hidden_output_size = 256
            regularizer = tf.contrib.layers.l2_regularizer(l2_reg_lambda)

            #regularizer = None
            with tf.variable_scope("projected_layer2", regularizer=regularizer):
                full_out = tf.contrib.layers.fully_connected(joined_feature, hidden_output_size,
                                                                activation_fn=tf.nn.relu,
                                                                reuse=False,
                                                                trainable=True,
                                                                scope="projected_layer")

            #full_out = tf.concat(axis=1, values=[full_out, self.extra_feature])
            last_weight_dim = full_out.get_shape()[1].value
            print("last_weight_dim: {}".format(last_weight_dim))
            bias = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
            s_w = tf.get_variable("s_w", shape=[last_weight_dim, 1], initializer=tf.contrib.layers.xavier_initializer())

            logits = tf.matmul(full_out, s_w) + bias

            print("logits shape")
            print(logits.get_shape())

            logits = tf.squeeze(logits, [1])

            self.probs = tf.sigmoid(logits, name="prob")

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=self.target)

            losses = tf.multiply(losses, self.target_loss_weight)

            self.mean_loss = tf.reduce_mean(losses, name="mean_loss") + l2_reg_lambda * l2_loss + sum(
                                                              tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.sign(self.probs - 0.5), tf.sign(self.target - 0.5))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name="accuracy")
