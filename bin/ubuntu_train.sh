cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data/Ubuntu_corpus_V2

answer_file=$DATA_DIR/answers.txt
train_file=$DATA_DIR/train.txt

embedded_vector_file=$DATA_DIR/glove_42B_300d_vec_plus_word2vec_100.txt

vocab_file=$DATA_DIR/vocab.txt
valid_file=$DATA_DIR/valid.txt
char_vocab_file=$DATA_DIR/char_vocab.txt

lambda=0

batch_size=128

dropout_keep_prob=1.0

DIM=400

max_word_length=18

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH python -u ${PKG_DIR}/answer_selection/train.py --answer_file $answer_file \
                --train_file $train_file \
                --embeded_vector_file $embedded_vector_file \
                --vocab_file $vocab_file \
                --valid_file $valid_file \
                --max_sequence_length 180 \
                --embedding_dim $DIM \
                --l2_reg_lambda $lambda \
                --dropout_keep_prob $dropout_keep_prob \
                --batch_size $batch_size \
                --rnn_size 200 \
                --evaluate_every 1000 \
                --char_vocab_file $char_vocab_file \
                --max_word_length $max_word_length
