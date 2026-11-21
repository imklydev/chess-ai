from process_data import tensor_move_to_tokenized_tensor_move, flatten_board
from model import model
from tokenizer import untokenize_move, tokenize_board
from bulletchess import Board, Move
from device import device

# For showing the board in a GUI instead of printing it
from webview import create_window, start
import time

import torch

def model_vs_model_chess_game(moves):
    print(f"The first {moves} moves of a Chess game, played by the model on both sides.")

    should_display_game = False if input("Should open a window to display the chess game visually? (Y/n): ") == 'n' else True

    board = Board()

    def board_to_html(board):

        from bulletchess import Board, Move, BLACK

        """Board HTML, always shown from White's perspective."""
        html = board._repr_html_()

        if board.turn is BLACK:
            # Rotate the board 180 deg so the perspective doesnt bounce from white to black
            html = (
                '<div style="display:inline-block; transform:scaleY(-1)">'
                + html
                + '</div>'
                + '<style>.bulletchess-board .row > div > div { transform: scaleY(-1); }'
                # Squares are now inverted, so invert their colors
                + '.bulletchess-board .lt { background-color: #444f61; }'
                + '.bulletchess-board .dk { background-color: #9aa8b4; }'
                + '</style>'
            )
        
        return html


    if(should_display_game):

        is_window_closed = False

        def on_closed():
            nonlocal is_window_closed
            is_window_closed = True

        page = (
            '<body style="margin:0">'
            f'<div id="b" style="display:inline-block">{board_to_html(board)}</div>'
            '</body>'
        )

        window = create_window(f"Model vs Model Chess game - Move {0}", html=page)

        window.events.closed += on_closed

    def play():
        nonlocal moves

        # Whites turn = 1
        turn = 1
        for i in range(moves):

            tokenized_board = torch.tensor(flatten_board([[turn]] + tokenize_board(board)), dtype=torch.float32).to(device)
            turn = 1 - turn

            move = untokenize_move(tensor_move_to_tokenized_tensor_move(model(tokenized_board)).tolist())

            board.apply(Move.from_uci(move))

            print(board)


            if should_display_game:
                window.set_title(f"Model vs Model Chess game - Move {i+1}")

                window.evaluate_js(
                    f"document.getElementById('b').innerHTML = `{board_to_html(board)}`"
                )

                if(is_window_closed):
                    return

                time.sleep(1.5)

    if should_display_game:
        # Block untill window is closed
        start(play)
    else:
        play()
    
