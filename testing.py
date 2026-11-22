from process_data import tensor_move_to_tokenized_tensor_move
from model import model
from tokenizer import untokenize_move, tokenize_board
from bulletchess import Board, Move
from device import device

import numpy as np

# For showing the board in a GUI instead of printing it
from webview import create_window, start
import time

import sys, os

import torch

from model import pick_best_predicted_legal_move

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
        
        def on_closed():
            print("Window closed, ending this test...")
            sys.stdout.flush()
            os._exit(0)

        page = (
            '<body style="margin:0">'
            f'<div id="b" style="display:inline-block">{board_to_html(board)}</div>'
            '</body>'
        )

        window = create_window(f"Model vs Model Chess game - Move {0}", html=page)

        window.events.closed += on_closed


    def play():
        nonlocal moves

        # Wait for the window to fully initialize
        time.sleep(1)


        # Whites turn = 1
        turn = 1
        for i in range(moves):

            turn_plane = np.full((1, 8, 8), turn, dtype=np.float32)  

            tokenized_board = torch.tensor(np.concatenate([tokenize_board(board), turn_plane]), dtype=torch.float32).to(device)

            # Make the board be of shape [1, 13, 8, 8] instead of [13, 8, 8]
            tokenized_board = tokenized_board.unsqueeze(0)

            turn = 1 - turn

            move = pick_best_predicted_legal_move(model(tokenized_board).cpu(), board)

            board.apply(Move.from_uci(move))

            print(board)



            if should_display_game:

                window.set_title(f"Model vs Model Chess game - Move {i+1}")

                window.evaluate_js(
                    f"document.getElementById('b').innerHTML = `{board_to_html(board)}`"
                )

                time.sleep(1)

    if should_display_game:
        # Block untill window is closed
        start(play)
    else:
        play()
    
