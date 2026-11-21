from download_training_data import get_training_data

training_data = get_training_data()

from tokenizer import tokenize_data, untokenize_move, untokenize_board

tokenized_boards, tokenized_moves = tokenize_data(training_data)

from process_data import get_inputs_and_outputs, flatten_inputs_and_outputs
import numpy as np

training_inputs, training_outputs = get_inputs_and_outputs(tokenized_boards, tokenized_moves)

print(untokenize_board(training_inputs[0]), untokenize_move(training_outputs[0]), sep="")

training_inputs, training_outputs = flatten_inputs_and_outputs(training_inputs, training_outputs)

import torch

training_inputs = torch.tensor(np.array(training_inputs), dtype=torch.float32)
training_outputs = torch.tensor(np.array(training_outputs), dtype=torch.float32)

from model import model
from process_data import tensor_move_to_tokenized_tensor_move

print(untokenize_move(tensor_move_to_tokenized_tensor_move(model(training_inputs[0])).tolist()))

from train import train_model

model = train_model(model, 100, training_inputs, training_outputs)

