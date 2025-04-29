from environment import reward, action_space
from attacks import attack_fn_map
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from collections import deque
import numpy as np
from tqdm import tqdm
import torch 

# 25 minutes - 1 episode of 1 time step


img_dim = 640  # Image dimension for YOLO model
patch_dim = 64  # Patch dimension for attacks

class RLAttackDataset(Dataset):
    def __init__(self, image_dir, img_dim=640):
        """
        Args:
            image_dir (str): Directory with all the images.
            img_dim (int): Size to which all images are resized (img_dim x img_dim).
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.img_dim = img_dim
        self.transform = transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),  # Converts image to [0, 1] and channels-first (C, H, W)
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Ensure RGB
        image_tensor = self.transform(image)
        return image_tensor, self.image_files[idx]  # Return filename for tracking
    

# Example usage:
image_dir = "C:/Users/prana/fiftyone/coco-2017/validation/data"
dataset = RLAttackDataset(image_dir=image_dir, img_dim=640)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, autoshape=False)
model.eval()
model.to(device)

class DQN(nn.Module):
    def __init__(self, action_dim):
        super(DQN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # [1, 32, 320, 320]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # [1, 64, 160, 160]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# [1, 128, 80, 80]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# [1, 256, 40, 40]
            nn.ReLU(),
        )
        
        self.fc = nn.Linear(256*1600, action_dim)

    def forward(self, x):
        x = self.conv_layers(x)  
        x = x.view(-1, 256 *1600) 
        x = self.fc(x)              # Final output: [1, n_dimensions]
        return x



# Hyperparameters
gamma = 0.99
epsilon = 0.1
lr = 1e-4
batch_size = 8
replay_buffer = deque(maxlen=10000)
num_episodes = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment setup
n_patches = (img_dim // patch_dim) ** 2
n_attacks = len(attack_fn_map)
action_space = [(i, j, k) for i in range(img_dim // patch_dim) for j in range(img_dim // patch_dim) for k in range(n_attacks)]
action_dim = len(action_space)

# Initialize DQN
policy_net = DQN(action_dim).to(device)
target_net = DQN(action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer = optim.Adam(policy_net.parameters(), lr=lr)


def select_action(state):
    
    if random.random() < epsilon:
        action_idx = random.randint(0, action_dim - 1)
        return action_idx
    
    with torch.no_grad():
        q_values = policy_net(state)
        action_idx = q_values.argmax().item()
        # print("q_values", q_values.shape)
        # print("action_idx", action_idx) 
        return action_idx

def optimize_model():
    if len(replay_buffer) < batch_size:
        return
    transitions = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states = zip(*transitions)

    states = torch.stack(states).to(device)
    next_states = torch.stack(next_states).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float).to(device)

    q_values = policy_net(states)
    next_q_values = target_net(next_states).detach()
    q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

    target_q = rewards + gamma * next_q_values.max(1)[0]
    loss = F.mse_loss(q_selected, target_q)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


for episode in tqdm(range(num_episodes)):
    
    for state_img, _ in tqdm(dataloader):
        
        state = state_img.squeeze(0).to(device)  # [C, H, W]
        
        action_idx = select_action(state)

        patch_x, patch_y, attack_idx = action_space[action_idx]
        
        # Apply attack
        next_state = attack_fn_map[attack_idx](state.cpu(), patch_x, patch_y)

        # Reward
        r = reward(state, next_state, model).item()
        
        # Store in buffer
        replay_buffer.append((state.detach(), action_idx, r, next_state.detach()))
        
        # Optimize DQN
        optimize_model()

    # Update target network every few episodes
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(f"Episode {episode}: target net updated.")
