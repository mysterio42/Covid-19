import numpy as np
import torch
import torch.nn as nn


class CoronaVirusPredictor(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers=2):
        super(CoronaVirusPredictor, self).__init__()

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.5
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)

    def reset_hidden_states(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        )

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred


def train_model(model: CoronaVirusPredictor,
                train_data: torch.Tensor, train_labels: torch.Tensor,
                test_data: torch.Tensor = None, test_labels: torch.Tensor = None,
                num_epochs: int = 60):
    loss_fn = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_states()
        y_pred = model(train_data)
        loss = loss_fn(y_pred.float(), train_labels)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(test_data)
                test_loss = loss_fn(y_test_pred.float(), test_labels)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print('Epoch {} train loss: {} test loss: {}'.
                      format(t, loss.item(), test_loss.item()))
        elif t % 10 == 0:
            print('Epoch {} train loss: {}'.
                  format(t, loss.item()))
        train_hist[t] = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model.eval(), train_hist, test_hist
