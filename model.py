import torch
import torch.nn as nn
from device import device

from process_data import tensor_move_to_tokenized_tensor_move

class ChessAI(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv2D = nn.Sequential(
            # 12 piece type channels + 1 turn channel
            nn.Conv2d(in_channels=13, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            # 16 output channels from self.conv2D of an 8*8 board
            nn.Linear(16*8*8, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # 32 outputs (8 start rows, 8 start cols, 8 end rows, 8 end cols)
            nn.Linear(256, 32),
        )


    def forward(self, x):
        return self.fc(self.conv2D(x))

def pick_best_predicted_legal_move(output: torch.Tensor, board) -> str:
    """Return the highest-scoring predicted legal move (as a UCI string) for the board"""

    import math

    # 32 outputs -> 4 groups of 8 (start col, start row, dest col, dest row)
    log_probs = torch.log_softmax(output.view(4, 8), dim=1).tolist()


    best_uci, best_score = None, -math.inf
    for m in board.legal_moves():
        uci = m.uci()
        score = (
            log_probs[0][ord(uci[0]) - ord('a')]   # start column
            + log_probs[1][int(uci[1]) - 1]        # start row
            + log_probs[2][ord(uci[2]) - ord('a')] # dest column
            + log_probs[3][int(uci[3]) - 1]        # dest row
        )

        # Promotions share the same 4 chars; prefer the queen as a tie-break
        if len(uci) == 5 and uci[4] == 'q':
            score += 1e-6
        
        if score > best_score:
            best_uci, best_score = uci, score
    
    return best_uci


model = ChessAI()
model = model.to(device)

def save_model(model):
    torch.save(model.state_dict(), "model.pth")

def load_model(model):
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    return model