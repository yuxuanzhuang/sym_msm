import torch
import torch.nn as nn
        
class MultimerNet(nn.Module):
    def __init__(self, data_shape, multimer, n_states):
        super().__init__()
        self.data_shape = data_shape
        self.multimer = multimer
        self.n_states = n_states

        self.n_feat_per_sub = self.data_shape // self.multimer
        self._construct_architecture()

    def _construct_architecture(self):
        self.batchnorm1d = nn.BatchNorm1d(self.n_feat_per_sub)

        # Fully connected layers into monomer part
        self.fc1 = nn.Linear(self.n_feat_per_sub, 200)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(200, 100)
        self.elu2 = nn.ELU()
        self.fc3 = nn.Linear(100, 50)
        self.elu3 = nn.ELU()
        self.fc4 = nn.Linear(50, 20)
        self.elu4 = nn.ELU()
        self.fc5 = nn.Linear(20, self.n_states)
        self.softmax = nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    # x represents our data
    def forward(self, x):
        batch_size = x.shape[0]
        n_feat_per_sub = int(self.data_shape / self.multimer)
        x_splits = x.reshape(batch_size, self.multimer, self.n_feat_per_sub)
        output = []
        x_stack = torch.permute(x_splits, (1, 0, 2)).reshape(
            batch_size * self.multimer, self.n_feat_per_sub)
        x_stack = self.batchnorm1d(x_stack)
        x_stack = self.fc1(x_stack)
        x_stack = self.elu1(x_stack)
        x_stack = self.dropout1(x_stack)
        x_stack = self.fc2(x_stack)
        x_stack = self.elu2(x_stack)
        x_stack = self.dropout2(x_stack)
        x_stack = self.fc3(x_stack)
        x_stack = self.elu3(x_stack)
        x_stack = self.fc4(x_stack)
        x_stack = self.elu4(x_stack)
        x_stack = self.fc5(x_stack)
        x_stack = self.softmax(x_stack)
        x_splits = x_stack.reshape(
            self.multimer,
            batch_size,
            self.n_states).permute(
            1,
            0,
            2).reshape(
            batch_size,
            self.n_states * self.multimer)
        return x_splits

class MultimerNet_200(MultimerNet):
    def _construct_architecture(self):
        self.batchnorm1d = nn.BatchNorm1d(self.n_feat_per_sub, dtype=torch.float32)

        # Fully connected layers into monomer part
        self.fc1 = nn.Linear(self.n_feat_per_sub, 200)
        self.elu1 = nn.ELU()

        self.fc2 = nn.Linear(200, 100)
        self.elu2 = nn.ELU()

        self.fc3 = nn.Linear(100, 50)
        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(50, 20)
        self.elu4 = nn.ELU()

        self.fc5 = nn.Linear(20, self.n_states)
        self.softmax = nn.Softmax(dim=1)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    # x represents our data
    def forward(self, x):
        batch_size = x.shape[0]

        n_feat_per_sub = int(self.data_shape / self.multimer)
        x_splits = x.reshape(batch_size, self.multimer, self.n_feat_per_sub)
        output = []

        x_stack = torch.permute(x_splits, (1, 0, 2)).reshape(
            batch_size * self.multimer, self.n_feat_per_sub
        )

        x_stack = self.batchnorm1d(x_stack)
        x_stack = self.fc1(x_stack)
        x_stack = self.elu1(x_stack)
        x_stack = self.dropout1(x_stack)
        x_stack = self.fc2(x_stack)
        x_stack = self.elu2(x_stack)
        x_stack = self.dropout2(x_stack)
        x_stack = self.fc3(x_stack)
        x_stack = self.elu3(x_stack)
        x_stack = self.fc4(x_stack)
        x_stack = self.elu4(x_stack)
        x_stack = self.fc5(x_stack)
        x_stack = self.softmax(x_stack)

        x_splits = (
            x_stack.reshape(self.multimer, batch_size, self.n_states)
            .permute(1, 0, 2)
            .reshape(batch_size, self.n_states * self.multimer)
        )
        return x_splits

class MultimerNet_400(MultimerNet):
    def _construct_architecture(self):
        self.batchnorm1d = nn.BatchNorm1d(self.n_feat_per_sub, dtype=torch.float32)

        # Fully connected layers into monomer part
        self.fc1 = nn.Linear(self.n_feat_per_sub, 400)
        self.elu1 = nn.ELU()

        self.fc2 = nn.Linear(400, 200)
        self.elu2 = nn.ELU()

        self.fc3 = nn.Linear(200, 50)
        self.elu3 = nn.ELU()

        self.fc4 = nn.Linear(50, 20)
        self.elu4 = nn.ELU()

        self.fc5 = nn.Linear(20, self.n_states)
        self.softmax = nn.Softmax(dim=1)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    # x represents our data
    def forward(self, x):
        batch_size = x.shape[0]

        n_feat_per_sub = int(self.data_shape / self.multimer)
        x_splits = x.reshape(batch_size, self.multimer, self.n_feat_per_sub)
        output = []

        x_stack = torch.permute(x_splits, (1, 0, 2)).reshape(
            batch_size * self.multimer, self.n_feat_per_sub
        )

        x_stack = self.batchnorm1d(x_stack)
        x_stack = self.fc1(x_stack)
        x_stack = self.elu1(x_stack)
        x_stack = self.dropout1(x_stack)
        x_stack = self.fc2(x_stack)
        x_stack = self.elu2(x_stack)
        x_stack = self.dropout2(x_stack)
        x_stack = self.fc3(x_stack)
        x_stack = self.elu3(x_stack)
        x_stack = self.fc4(x_stack)
        x_stack = self.elu4(x_stack)
        x_stack = self.fc5(x_stack)
        x_stack = self.softmax(x_stack)

        x_splits = (
            x_stack.reshape(self.multimer, batch_size, self.n_states)
            .permute(1, 0, 2)
            .reshape(batch_size, self.n_states * self.multimer)
        )
        return x_splits