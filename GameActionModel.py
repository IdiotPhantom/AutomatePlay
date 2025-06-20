import torch
import torch.nn as nn
import os
from setting import NN_FILE_NAME
from torchvision.models import resnet18, ResNet18_Weights


class GameActionModel(nn.Module):
    def __init__(self, num_tasks):
        super(GameActionModel, self).__init__()

        # CNN feature extractor
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.cnn.fc = nn.Identity()

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, 32)

        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 4),   # ⬅ Add 4th output: confidence
            nn.Sigmoid()         # Optional: squash all outputs between 0–1
        )

        # Load if exists
        if os.path.exists(NN_FILE_NAME):
            self.load_state_dict(torch.load(NN_FILE_NAME))
            self.eval()

    def forward(self, image_tensor, task_id):
        image_feat = self.cnn(image_tensor)
        task_feat = self.task_embedding(task_id)
        combined = torch.cat([image_feat, task_feat], dim=1)
        return self.fc(combined)

    def save(self):
        torch.save(self.state_dict(), NN_FILE_NAME)
