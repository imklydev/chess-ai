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

def untokenize_move(move):
    one_hot_start_cols = move[:8]
    one_hot_start_rows = move[8:16]
    one_hot_end_cols = move[16:24]
    one_hot_end_rows = move[24:32]

    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    start_col = cols[one_hot_start_cols.index(1)]
    start_row = one_hot_start_rows.index(1) + 1
    end_col = cols[one_hot_end_cols.index(1)]
    end_row = one_hot_end_rows.index(1) + 1

    return start_col + str(start_row) + end_col + str(end_row)

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

def untokenize_board(tokenized_board):

    # Drop turn
    tokenized_board = tokenized_board[1:]

    from bulletchess import WHITE, BLACK, PAWN, BISHOP, KNIGHT, ROOK, QUEEN, KING
    from bulletchess import Piece, SQUARES

    board = Board.empty()

    piece_types = [
        (color, piece)
        for color in (WHITE, BLACK)
        for piece in (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
    ]

    # 12 different piece types
    for i in range(12):

        piece_indexes = [idx for idx, x in enumerate(tokenized_board[i]) if x == 1]

        for piece_index in piece_indexes:
            board[SQUARES[piece_index]] = Piece(piece_types[i][0], piece_types[i][1])

    return board



def tokenize_data(games):
    """
    Returns tokenized_boards and tokenized_moves as lists
    """

    tokenized_moves = []
    tokenized_boards = []

    for game in games:

        tokenized_game = []
        tokenized_boards_for_this_game = []

        board = Board()

        # Black to move: turn = 0
        # White to move: turn = 1
        turn = 0
        for move in game:
            turn = 1 - turn

            # SAN format move (Nf3, d4, etc.)
            san_move = Move.from_san(move, board)

            # Add the board state BEFORE applying the move, so the inputs and outputs make sense when training
            # Also add a (turn) variable
            tokenized_boards_for_this_game.append([turn] + tokenize_board(board))


            board.apply(san_move)

            # UCI format move (d2d4, etc.)
            move = Move.uci(san_move)

            tokenized_game.append(tokenize_move(move))



        tokenized_moves.append(tokenized_game)
        tokenized_boards.append(tokenized_boards_for_this_game)

    return tokenized_boards, tokenized_moves

