from download_training_data import get_training_data

training_data = get_training_data()

from tokenizer import tokenize_data, untokenize_move, untokenize_board

tokenized_boards, tokenized_moves = tokenize_data(training_data)

from process_data import get_inputs_and_outputs, flatten_inputs_and_outputs
import numpy as np

training_inputs, training_outputs = get_inputs_and_outputs(tokenized_boards, tokenized_moves)

training_inputs, training_outputs = flatten_inputs_and_outputs(training_inputs, training_outputs)

import torch
from device import device

training_inputs = torch.tensor(np.array(training_inputs), dtype=torch.float32).to(device)
training_outputs = torch.tensor(np.array(training_outputs), dtype=torch.float32).to(device)

from model import model, save_model, load_model
from process_data import tensor_move_to_tokenized_tensor_move

from train import train_model

model = train_model(model, 100, training_inputs, training_outputs)

save_model(model)

print(untokenize_move(tensor_move_to_tokenized_tensor_move(model(training_inputs[0])).tolist()))