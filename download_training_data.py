import requests
import os
import zstandard as zstd

def get_training_data():
    if not os.path.exists("training_data.txt"):

        # Download chess game data
        r = requests.get("https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst")

        # Unzip downloaded data
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(r.content, read_across_frames=True) as reader:
            decompressed_data = reader.read()

        # Write downloaded data
        with open("training_data.txt", "wb") as f:
            f.write(decompressed_data)
    else:

        with open("training_data.txt", "r") as f:
            decompressed_data = f.read()

    # Use 150k chars from the data (not sure how many games this would be, maybe like <1k ?)
    decompressed_data = decompressed_data[:150000]

    games = []

    for line in decompressed_data.splitlines():
        if line.startswith("[") or len(line) < 2:
            continue

        moves_and_move_numbers = line.split(" ")

        moves_of_this_game = []

        drop_game = False

        for move in moves_and_move_numbers:

            # This game has evaluation included, which makes it kinda harder
            # to extract moves, so drop it
            if(move == '{'):
                drop_game = True
                break

            # This "move" is a move number
            if(move[-1] == '.'):
                continue

            # This move is the last move
            if(move[-1] == '#'):

                # Remove '#'
                move = move[:-1]

            moves_of_this_game.append(move)

        if(drop_game):
            continue

        # Remove result
        moves_of_this_game.pop()


        games.append(moves_of_this_game)


    return games

