
def get_inputs_and_outputs(tokenized_boards, tokenized_moves):
    inputs = []
    outputs = []

    # tokenized_boards and tokenized_moves is of shape [games][game_length] ...
    games = len(tokenized_boards)

    for i in range(games):
        game_length = len(tokenized_boards[i])
        for j in range(game_length):
            inputs.append(tokenized_boards[i][j])
            outputs.append(tokenized_moves[i][j])

    return inputs, outputs

def flatten_board(board):

    import numpy as np

    flattened_board = np.concatenate([np.asarray(r) for r in board])

    return flattened_board

def flatten_inputs_and_outputs(inputs, outputs):

    import numpy as np

    flattened_inputs = np.stack([
        np.concatenate([np.asarray(r) for r in sample])
        for sample in inputs
    ])

    return flattened_inputs, outputs

def tensor_move_to_tokenized_tensor_move(tensor_move):
    import torch

    start_col = torch.argmax(tensor_move[:8]).item()
    start_row = torch.argmax(tensor_move[8:16]).item()
    end_col = torch.argmax(tensor_move[16:24]).item()
    end_row = torch.argmax(tensor_move[24:32]).item()

    one_hot_start_cols = [0] * 8
    one_hot_start_cols[start_col] = 1

    one_hot_start_rows = [0] * 8
    one_hot_start_rows[start_row] = 1

    one_hot_end_cols = [0] * 8
    one_hot_end_cols[end_col] = 1

    one_hot_end_rows = [0] * 8
    one_hot_end_rows[end_row] = 1

    move = one_hot_start_cols + one_hot_start_rows + one_hot_end_cols + one_hot_end_rows

    return torch.tensor(move)