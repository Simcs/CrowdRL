import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, shape_int, shape_ext, num_outputs, std=0.0):
        super(ActorCritic, self).__init__()

        hidden_size1 = 256
        hidden_size2 = 128
        hidden_size3 = 64

        hidden_size1 = 32
        hidden_size2 = 64
        hidden_size3 = 96

        conv_size = 16
        filter_size = 3

        height = shape_ext[0]
        width = shape_ext[1]
        flatten_size = (height - filter_size + 1) * (width - filter_size + 1) * conv_size

        self.critic_int = nn.Sequential(
            nn.Linear(shape_int, hidden_size1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
        )

        self.critic_ext = nn.Sequential(
            nn.Conv2d(1, conv_size, kernel_size=filter_size, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(flatten_size, hidden_size1),
        )

        self.critic_final = nn.Sequential(
            nn.Linear(hidden_size3, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, 1),
        )

        self.actor_int = nn.Sequential(
            nn.Linear(shape_int, hidden_size1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
        )

        self.actor_ext = nn.Sequential(
            nn.Conv2d(1, conv_size, kernel_size=filter_size, stride=1, padding=0),
            nn.Flatten(),
            nn.Linear(flatten_size, hidden_size1),
        )

        self.actor_final = nn.Sequential(
            nn.Linear(hidden_size3, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, num_outputs),
        )


        # self.critic_cnn = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=1),
        # )

        # self.critic = nn.Sequential(
        #     nn.Linear
        # )
        
        # self.critic = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size1),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size1, hidden_size2),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size2, hidden_size3),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size3, 1),
        # )
        
        # self.actor = nn.Sequential(
        #     nn.Linear(num_inputs, hidden_size1),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size1, hidden_size2),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size2, hidden_size3),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size3, num_outputs),
        # )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
    def forward(self, int_state, ext_state):

        # int_state = torch.FloatTensor(state[0]).to(device)
        # ext_state = torch.FloatTensor(state[1]).unsqueeze(1).to(device)

        value_int = self.critic_int(int_state)
        value_ext = self.critic_ext(ext_state)
        value_cat = torch.cat((value_int, value_ext), axis=1)
        value = self.critic_final(value_cat)

        mu_int    = self.actor_int(int_state)
        mu_ext    = self.actor_ext(ext_state)
        mu_cat    = torch.cat((mu_int, mu_ext), axis=1)
        mu        = self.actor_final(mu_cat)
        std   = self.log_std.exp().expand_as(mu)

        # value = self.critic(x)
        # mu    = self.actor(x)
        dist  = Normal(mu, std)
        return dist, value
