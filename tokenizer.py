from bulletchess import Board, Move

from download_training_data import get_training_data

def tokenize_move(move):
    """
    Move format is UCI (d2d4, e2e4, etc.)\n
    Returns one-hot encoded columns and rows for start and end positions (concatinated)
    """

    tokenized_move = []

    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    start_col = move[0]
    start_row = move[1]
    end_col = move[2]
    end_row = move[3]

    # Transform everything into ints
    start_col = cols.index(start_col)
    start_row = int(start_row)-1
    end_col = cols.index(end_col)
    end_row = int(end_row)-1

    one_hot_start_cols = [0] * 8
    one_hot_start_cols[start_col] = 1

    one_hot_start_rows = [0] * 8
    one_hot_start_rows[start_row] = 1

    one_hot_end_cols = [0] * 8
    one_hot_end_cols[end_col] = 1

    one_hot_end_rows = [0] * 8
    one_hot_end_rows[end_row] = 1


    return one_hot_start_cols + one_hot_start_rows + one_hot_end_cols + one_hot_end_rows

def tokenize_board(board):

    from bulletchess import WHITE, BLACK, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING
    from bulletchess import  SQUARES

    SQUARE_INDEX = {sq: i for i, sq in enumerate(SQUARES)}

    piece_boards = []

    for color in (WHITE, BLACK):
        for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING):

            # Create a 8*8 board for each type of piece, and make the squares
            # the pieces are on 1

            piece_board = [0] * 64
            for square in board[color, piece]:
                piece_board[SQUARE_INDEX[square]] = 1

            piece_boards.append(piece_board)

    return piece_boards

def tokenize_data(games):
    """
    Returns tokenized_boards and tokenized_moves as lists
    """

    tokenized_moves = []
    tokenized_boards = []

    for game in games:

        tokenized_game = []

        board = Board()

        for move in game:

            # SAN format move (Nf3, d4, etc.)
            san_move = Move.from_san(move, board)

            board.apply(san_move)

            # UCI format move (d2d4, etc.)
            move = Move.uci(san_move)

            tokenized_game.append(tokenize_move(move))

            tokenized_boards.append(tokenize_board(board))


        tokenized_moves.append(tokenized_game)

    return tokenized_boards, tokenized_moves

