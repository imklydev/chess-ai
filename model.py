import torch
import torch.nn as nn

from process_data import tensor_move_to_tokenized_tensor_move

class ChessAI(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            # 12 piece types, each with an 8*8 board + 1 turn input
            nn.Linear(12*8*8+1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # 32 outputs (8 start rows, 8 start cols, 8 end rows, 8 end cols)
            nn.Linear(512, 32),
        )


    def forward(self, x):
        return tensor_move_to_tokenized_tensor_move(self.model(x))

model = ChessAI()
