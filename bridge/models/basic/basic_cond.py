import torch
from .layers import MLP
from .time_embedding import get_timestep_embedding

class ScoreNetworkCond(torch.nn.Module):

    def __init__(self, encoder_layers=[16], pos_dim=16, decoder_layers=[128,128], x_dim=1):
        super().__init__()
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim *2
        self.locals = [encoder_layers, pos_dim, decoder_layers, x_dim]

        self.net = MLP(3 * t_enc_dim,
                       layer_widths=decoder_layers +[x_dim],
                       activate_final = False,
                       activation_fn=torch.nn.LeakyReLU())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.x_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())

        self.y_encoder = MLP(x_dim,
                             layer_widths=encoder_layers +[t_enc_dim],
                             activate_final = False,
                             activation_fn=torch.nn.LeakyReLU())        

    def forward(self, x, y, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y.shape) == 1:
            x = y.unsqueeze(0)
            
        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        yemb = self.x_encoder(y)
        h = torch.cat([xemb , yemb, temb], -1)
        out = self.net(h) 
        return out
