import torch.nn as nn
import torch

class NCF(nn.Module):
    def __init__(self, num_users, num_movies, latent_dim, hidden_dims, dropout_rate=0.3, use_batchnorm=True):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.movie_embedding = nn.Embedding(num_movies, latent_dim)
        
        layers = []
        input_dim = 2 * latent_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(input_dim, hdim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hdim
        
        layers.append(nn.Linear(input_dim, 1))
        self.fc = nn.Sequential(*layers)
        self.global_bias = nn.Parameter(torch.tensor([0.0]))

    def forward(self, user_ids, movie_ids):
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        x = torch.cat([user_emb, movie_emb], dim=1)
        output = self.fc(x)
        return output.squeeze() + self.global_bias
