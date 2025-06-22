# === GameActionModel.py (refactored version) ===
import torch
import torch.nn as nn
import os
from typing import cast
from setting import NN_FILE_NAME
from torchvision.models import resnet18, ResNet18_Weights


class GameActionModel(nn.Module):
    def __init__(self, num_tasks):
        super(GameActionModel, self).__init__()

        # Load pretrained ResNet18 and adapt for grayscale
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self._convert_to_grayscale()

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, 32)

        # MLP head for predicting x, y, done, confidence
        self.mlp = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        # Load pretrained model if exists
        if os.path.exists(NN_FILE_NAME):
            state_dict = torch.load(NN_FILE_NAME)
            renamed_state_dict = {}
            for key in state_dict:
                new_key = key.replace("fc", "mlp")
                renamed_state_dict[new_key] = state_dict[key]
            missing_keys, unexpected_keys = self.load_state_dict(
                renamed_state_dict, strict=False)
            if missing_keys:
                print(f"[!] Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"[!] Unexpected keys: {unexpected_keys}")

    def _convert_to_grayscale(self):
        # Modify the first conv layer to accept grayscale images
        orig_conv1 = self.cnn.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv1.out_channels,
            kernel_size=cast(tuple[int, int], orig_conv1.kernel_size),
            stride=cast(tuple[int, int], orig_conv1.stride),
            padding=cast(tuple[int, int], orig_conv1.padding),
            bias=orig_conv1.bias is not None
        )

        with torch.no_grad():
            new_conv1.weight.copy_(orig_conv1.weight.mean(dim=1, keepdim=True))
        self.cnn.conv1 = new_conv1
        self.cnn.fc = nn.Identity()  # type: ignore # Remove final classifier

    def forward(self, image_tensor, task_id):
        assert image_tensor.dtype == torch.float32
        assert task_id.dtype == torch.long
        assert image_tensor.device == task_id.device

        image_feat = self.cnn(image_tensor)  # [batch, 512]
        task_feat = self.task_embedding(task_id)  # [batch, 32]

        combined = torch.cat([image_feat, task_feat], dim=1)
        output = self.mlp(combined)  # [batch, 4]

        # normalized x, y in [0, 1]
        x_y = torch.sigmoid(output[:, :2])
        # raw logits for BCEWithLogitsLoss
        done_score = output[:, 2]
        confidence = torch.sigmoid(output[:, 3])      # confidence in [0, 1]

        return x_y, done_score, confidence

    def save(self):
        torch.save(self.state_dict(), NN_FILE_NAME)
