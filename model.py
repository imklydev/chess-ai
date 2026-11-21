import torch
import torch.nn as nn
from device import device

from process_data import tensor_move_to_tokenized_tensor_move

class ChessAI(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = nn.Sequential(
            # 12 piece types, each with an 8*8 board + 1 turn input
            nn.Linear(12*8*8+1, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # 32 outputs (8 start rows, 8 start cols, 8 end rows, 8 end cols)
            nn.Linear(256, 32),
        )


    def forward(self, x):
        return self.model(x)

model = ChessAI()
model = model.to(device)

def save_model(model):
    torch.save(model.state_dict(), "model.pth")

def load_model(model):
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    return model