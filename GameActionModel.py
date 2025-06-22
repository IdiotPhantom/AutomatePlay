import torch
import torch.nn as nn
import os
from setting import NN_FILE_NAME
from torchvision.models import resnet18, ResNet18_Weights


class GameActionModel(nn.Module):
    def __init__(self, num_tasks):
        super(GameActionModel, self).__init__()

        # Load pretrained resnet18
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()  # Remove final classifier

        # === Modify conv1 to accept 1-channel grayscale input ===
        orig_conv1 = self.cnn.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv1.out_channels,
            kernel_size=orig_conv1.kernel_size,
            stride=orig_conv1.stride,
            padding=orig_conv1.padding,
            bias=orig_conv1.bias is not None
        )
        # Average pretrained RGB weights to init grayscale conv
        new_conv1.weight.data = orig_conv1.weight.data.mean(dim=1, keepdim=True)
        self.cnn.conv1 = new_conv1

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, 32)

        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(512 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4))

        # Load model if available
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

    def forward(self, image_tensor, task_id):
        assert image_tensor.dtype == torch.float32
        assert task_id.dtype == torch.long
        assert image_tensor.device == task_id.device

        image_feat = self.cnn(image_tensor)  # [batch, 512]
        task_feat = self.task_embedding(task_id)  # [batch, 32]

        combined = torch.cat([image_feat, task_feat], dim=1)
        output = self.mlp(combined)  # [batch, 4]

        x_y = torch.sigmoid(output[:, :2])  # normalized coords
        done_score = output[:, 2]           # raw logits
        confidence = torch.sigmoid(output[:, 3])

        return torch.cat([x_y, done_score.unsqueeze(1), confidence.unsqueeze(1)], dim=1)

    def save(self):
        torch.save(self.state_dict(), NN_FILE_NAME)
