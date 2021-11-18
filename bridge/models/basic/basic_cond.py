import torch
from .layers import MLP
from .time_embedding import get_timestep_embedding

class ScoreNetworkCond(torch.nn.Module):

    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=1, y_dim=1, time_embedding_denom=10000):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim, y_dim]

        self.net = MLP(3 * t_enc_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.xy_encoder = MLP(x_dim + y_dim,
                              layer_widths=[enc_dim*2 for enc_dim in encoder_layers] + [2*t_enc_dim],
                              activate_final = False,
                              activation_fn=torch.nn.LeakyReLU())

        self.time_embedding_denom = time_embedding_denom

    def forward(self, x, y, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y.shape) == 1:
            y = y.unsqueeze(0)

        t_emb = get_timestep_embedding(t, self.temb_dim, self.time_embedding_denom)
        t_emb = self.t_encoder(t_emb)
        xy_emb = self.xy_encoder(torch.cat([x, y], -1))
        h = torch.cat([xy_emb, t_emb], -1)
        out = self.net(h) 
        return out
