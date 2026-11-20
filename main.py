from download_training_data import get_training_data

training_data = get_training_data()

from tokenizer import tokenize_data, untokenize_move

tokenized_boards, tokenized_moves = tokenize_data(training_data)

from process_data import get_inputs_and_outputs

training_inputs, training_outputs = get_inputs_and_outputs(tokenized_boards, tokenized_moves)

print(training_inputs[0], "\n", untokenize_move(training_outputs[0]), sep="")