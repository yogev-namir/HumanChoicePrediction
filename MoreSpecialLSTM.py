import torch
import torch.nn as nn
from utils.usersvectors import UsersVectors
import torch.nn.functional as F
from consts import *



class MoreSpecialLSTM(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dropout, logsoftmax=True, input_twice=False):
        super().__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.input_twice = input_twice
        self.Q_linear_layer = nn.Linear(hidden_dim, hidden_dim)
        self.K_linear_layer = nn.Linear(hidden_dim, hidden_dim)

        
        self.input_fc = nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(input_dim * 2, self.hidden_dim),
                                      nn.Dropout(dropout),
                                      nn.ReLU())

        self.main_task = nn.LSTM(input_size=self.hidden_dim,
                                 hidden_size=self.hidden_dim,
                                 batch_first=True,
                                 num_layers=self.n_layers,
                                 dropout=dropout)

        seq = [nn.Linear(self.hidden_dim + (input_dim if self.input_twice else 0), self.hidden_dim // 2),
               nn.ReLU(),
               nn.Linear(self.hidden_dim // 2, self.output_dim)]
        if logsoftmax:
            seq += [nn.LogSoftmax(dim=-1)]

        self.output_fc = nn.Sequential(*seq)

        self.user_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)
        self.game_vectors = UsersVectors(user_dim=self.hidden_dim, n_layers=self.n_layers)

    def get_self_attention_scores(self, hidden_dim, matrix):
        # Create dummy Query and Key matrices
        Q = matrix[:,-1:,:] 
        K = matrix

        # Pass Q and K through their respective linear layers
        Q = self.Q_linear_layer(Q)
        K = self.K_linear_layer(K)

        # Compute dot product of Q and K^T
        dot_product = torch.bmm(Q, K.transpose(1, 2))  # bmm for batch matrix multiplication 

        # Scaling factor (dimension of the keys)
        sqrt_dk = torch.sqrt(torch.tensor(hidden_dim).float())

        # Scale the dot product by sqrt(dk)
        scaled_dot_product = dot_product / sqrt_dk

        # Apply softmax along the last dimension (over keys)
        attention_scores = F.softmax(scaled_dot_product, dim=-1) 

        return attention_scores
    
    def init_game(self, batch_size=1):
        return torch.stack([self.game_vectors.init_user] * batch_size, dim=0)

    def init_user(self, batch_size=1):
        return torch.stack([self.user_vectors.init_user] * batch_size, dim=0)


    def forward(self, input_vec, game_vector, user_vector):
        lstm_input = self.input_fc(input_vec)
        # lstm_input = lstm_input.reshape(-1, self.hidden_dim)
        # output = self.output_fc(lstm_input)
        lstm_shape = lstm_input.shape
        shape = user_vector.shape
        assert game_vector.shape == shape
        if len(lstm_shape) != len(shape):
            lstm_input = lstm_input.reshape((1,) * (len(shape) - 1) + lstm_input.shape)
        user_vector = user_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        game_vector = game_vector.reshape(shape[:-1][::-1] + (shape[-1],))
        
        # Placeholder for list of tensors representing results from each round
        results_per_round = []
        attention_scores_list = []

        for round in range(DATA_ROUNDS_PER_GAME):
            relevant_rounds = lstm_input[:,:round+1,:] #relvant_rounds.shape = [batch, subgame_length, embedding]
            hidden_states_list = list()

            for history_length in range(round+1):  
                relvant_histories = relevant_rounds[:, history_length:, :] if history_length > 1 else relevant_rounds 
                lstm_output, (game_vector, user_vector) = self.main_task(relvant_histories.contiguous(),
                                                                 (game_vector.contiguous(),
                                                                  user_vector.contiguous())) # [batch, history_length, embedding]
                hidden_states_list.append(lstm_output[:,-1,:]) # each is [batch, embedding]

            hidden_states_tensor = torch.stack(hidden_states_list, dim=1) # [batch_size, num_hidden_states, hidden_size]
            attention_scores = self.get_self_attention_scores(hidden_dim=self.hidden_dim,
                                                         matrix=hidden_states_tensor) # [batch_size, 1, subgame_length]
           
            V = self.output_fc(hidden_states_tensor) # [batch_size, subgame_length, 2] 
            weighted_round_result = torch.bmm(attention_scores, V) # [batch_size, 1, 2] 

            results_per_round.append(weighted_round_result)

            # Pad attention scores to [batch size, 1, DATA_ROUNDS_PER_GAME]
            padding_size = DATA_ROUNDS_PER_GAME - (round + 1)
            if padding_size > 0:
                attention_scores = torch.nn.functional.pad(attention_scores, (0, padding_size), value=-1) # [batch_size, 1, DATA_ROUNDS_PER_GAME]
            
            attention_scores_list.append(attention_scores) # Store the attention scores


        # Stack all the results across the rounds to get a tensor of [batch_size, number of rounds, 2]
        output = torch.cat(results_per_round, dim=1)  # Use torch.cat to concatenate along the second dimension
        # Stack the attention scores along the batch dimension
        attention_scores_tensor = torch.cat(attention_scores_list, dim=1) # [batch_size * DATA_ROUNDS_PER_GAME, 1, DATA_ROUNDS_PER_GAME]

        user_vector = user_vector.reshape(shape)
        game_vector = game_vector.reshape(shape)
        # if hasattr(self, "input_twice") and self.input_twice:
        #     lstm_output = torch.cat([lstm_output, input_vec], dim=-1)
        # output = self.output_fc(lstm_output)
        # if len(output.shape) != len(lstm_shape):
        #     output.reshape(-1, output.shape[-1])
        if self.training:
            return {"output": output, "attention_scores": attention_scores_tensor, "game_vector": game_vector, "user_vector": user_vector}
        else:
            return {"output": output, "attention_scores": attention_scores_tensor, "game_vector": game_vector.detach(), "user_vector": user_vector.detach()}

    