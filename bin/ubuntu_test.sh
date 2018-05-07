cur_dir=`pwd`
parentdir="$(dirname $cur_dir)"

DATA_DIR=${parentdir}/data/Ubuntu_corpus_V2

latest_run=`ls -dt runs/* |head -n 1`
latest_checkpoint=${latest_run}/checkpoints
echo $latest_checkpoint

answer_file=$DATA_DIR/answers.txt
test_file=$DATA_DIR/test.txt
vocab_file=$DATA_DIR/vocab.txt
output_file=./ubuntu_test_out.txt
char_vocab_file=$DATA_DIR/char_vocab.txt
max_word_length=18

batch_size=128
max_sequence_length=180

PKG_DIR=${parentdir}

PYTHONPATH=${PKG_DIR}:$PYTHONPATH python -u ${PKG_DIR}/answer_selection/eval.py --answer_file $answer_file \
                  --test_file $test_file \
                  --vocab_file $vocab_file \
                  --max_sequence_length $max_sequence_length \
                  --batch_size $batch_size \
                  --output_file $output_file \
                  --checkpoint_dir $latest_checkpoint \
                  --char_vocab_file $char_vocab_file \
                  --max_word_length $max_word_length
