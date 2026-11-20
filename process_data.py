
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