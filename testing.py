from process_data import tensor_move_to_tokenized_tensor_move
from model import model
from tokenizer import untokenize_move, tokenize_board
from bulletchess import Board, Move
from device import device
from bulletchess import CHECKMATE, DRAW

import numpy as np

# For showing the board in a GUI instead of printing it
from webview import create_window, start
import time

import sys, os

from model import pick_best_predicted_legal_move

import torch

def board_to_html(board):
    """Board HTML, always shown from White's perspective."""

    from bulletchess import Board, Move, BLACK

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

def model_vs_model_chess_game(moves):
    print(f"The first {moves} moves of a Chess game, played by the model on both sides.")

    should_display_game = False if input("Should open a window to display the chess game visually? (Y/n): ") == 'n' else True

    board = Board()


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
    
def player_vs_model_chess_game():
    print(f"Player vs Model chess game test")

    board = Board()

    JS_FOR_PLAYER_CLICKING = """
    <script>
    let selected = null;

    document.addEventListener('click', (e) => {
        const sq = e.target.closest('.bulletchess-board .row > div');
        if (!sq) return;

        const row = sq.parentElement;
        const r = [...row.parentElement.querySelectorAll('.row')].indexOf(row);
        const c = [...row.children].indexOf(sq);

        if (selected === null) {
            selected = {r, c, el: sq};
            sq.style.outline = '3px solid #e8c547';
            sq.style.outlineOffset = '-3px';
        } else {
            const from = selected;
            from.el.style.outline = '';
            selected = null;

            // Clicking the same square again just deselects
            if (from.r === r && from.c === c) return;

            pywebview.api.on_player_move(from.c, from.r, c, r);
        }
    });
    </script>
    """

    page = (
        '<body style="margin:0">'
        + f'<div id="b" style="display:inline-block">{board_to_html(board)}</div>'
        + JS_FOR_PLAYER_CLICKING
        + '</body>'
    )

    window = create_window(f"Player vs Model chess game", html=page)

        
    def on_closed():
        print("Window closed, ending this test...")
        sys.stdout.flush()
        os._exit(0)

    window.events.closed += on_closed

    from queue import Queue

    player_moves = Queue()

    def player_move_to_square(col, row):
        """Returns a string with a square (e4, g7, a5...) from a column and row"""

        return "abcdefgh"[col] + str(8 - row) # (8 - row) because when getting the row in javascript we get a number from 0 to 7, 0 being the top row

    def on_player_move(from_col, from_row, to_col, to_row):
        player_moves.put(player_move_to_square(from_col, from_row) + player_move_to_square(to_col, to_row))

    window.expose(on_player_move)


    def play():

        # Wait for the window to fully initialize
        time.sleep(1)


        # Whites turn = 1
        turn = 1
        while True:
            # Wait for a player move 
            uci_player_move = player_moves.get()

            try:
                player_move = Move.from_uci(uci_player_move)
            except Exception: # Malformed move
                continue 

            if not player_move in board.legal_moves():

                try:
                    # Try adding a 'promote to queen' symbol to the player move, in case that was the problem (this also forces you to promote to a queen when promoting a pawn)
                    player_move = Move.from_uci(uci_player_move + 'q')
                except Exception:
                    continue # Malformed move

                if not player_move in board.legal_moves():
                    continue # Ilegal move

            board.apply(player_move)
            turn = 1 - turn

            # Update rendered board
            window.evaluate_js(
                f"document.getElementById('b').innerHTML = `{board_to_html(board)}`"
            )

            if(board in CHECKMATE) :
                print("You checkmated the model!")
                break
            if(board in DRAW):
                print("You drew with the model!")
                break

            turn_plane = np.full((1, 8, 8), turn, dtype=np.float32)  

            tokenized_board = torch.tensor(np.concatenate([tokenize_board(board), turn_plane]), dtype=torch.float32).to(device)

            # Make the board be of shape [1, 13, 8, 8] instead of [13, 8, 8]
            tokenized_board = tokenized_board.unsqueeze(0)

            move = pick_best_predicted_legal_move(model(tokenized_board).cpu(), board)

            board.apply(Move.from_uci(move))
            turn = 1 - turn

            # Update rendered board
            window.evaluate_js(
                f"document.getElementById('b').innerHTML = `{board_to_html(board)}`"
            )

            if(board in CHECKMATE) :
                print("The model checkmated you!")
                break
            if(board in DRAW):
                print("The model drew with you")
                break


    start(play)