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
        self.cnn.fc = nn.Identity()  # output: [batch, 512]

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, 32)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            # outputs: [x_norm, y_norm, done_score, confidence]
            nn.Linear(128, 4)
        )

        if os.path.exists(NN_FILE_NAME):
            state_dict = torch.load(NN_FILE_NAME)

            # Rename old keys to new ones
            renamed_state_dict = {}
            for key in state_dict:
                new_key = key.replace("fc", "mlp")  # Rename fc.* → mlp.*
                renamed_state_dict[new_key] = state_dict[key]

            # Load into the model
            self.eval()

    def forward(self, image_tensor, task_id):
        image_feat = self.cnn(image_tensor)  # [batch, 512]
        task_feat = self.task_embedding(task_id)  # [batch, 32]

        combined = torch.cat([image_feat, task_feat], dim=1)
        output = self.mlp(combined)  # [batch, 4]

        # Interpret outputs
        # normalized (0–1)
        x_y = torch.sigmoid(output[:, :2])
        done_score = output[:, 2]                               # raw (logits)
        # probability (0–1)
        confidence = torch.sigmoid(output[:, 3])

        return torch.cat([x_y, done_score.unsqueeze(1), confidence.unsqueeze(1)], dim=1)

    def save(self):
        torch.save(self.state_dict(), NN_FILE_NAME)
