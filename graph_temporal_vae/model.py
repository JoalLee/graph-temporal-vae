import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as parametrizations

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # First layer
        self.conv1 = parametrizations.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second layer
        self.conv2 = parametrizations.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return self.net(x)

class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns to weight time steps.
    Replaces Global Average Pooling with learnable attention weights.
    """
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x: [Batch, Hidden, Window]
        x = x.permute(0, 2, 1)  # [Batch, Window, Hidden]
        
        # Compute attention scores
        scores = self.attention(x)  # [Batch, Window, 1]
        weights = F.softmax(scores, dim=1)  # Normalize over time dimension
        
        # Weighted sum
        pooled = (x * weights).sum(dim=1)  # [Batch, Hidden]
        return pooled, weights  # Return weights for visualization

class Encoder(nn.Module):
    """
    Encoder with Attention Pooling for Global Latent Vector output.
    Input: [Batch, Features, Window]
    Output: mu, logvar each [Batch, Latent_Dim]
    """
    def __init__(self, input_dim, num_channels, kernel_size, dropout, latent_dim, window_size, dilations):
        super(Encoder, self).__init__()
        
        layers = []
        num_levels = len(dilations)
        self.num_channels = num_channels
        self.window_size = window_size
        self.latent_dim = latent_dim
        
        # Initial projection to hidden channels
        self.input_conv = nn.Conv1d(input_dim, num_channels, kernel_size=1)
        
        for i in range(num_levels):
            dilation_size = dilations[i]
            pad = (kernel_size - 1) * dilation_size // 2
            
            layers.append(
                TemporalBlock(
                    n_inputs=num_channels, 
                    n_outputs=num_channels, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    dilation=dilation_size, 
                    padding=pad, 
                    dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)
        
        # Attention Pooling (replaces Global Average Pooling)
        self.attention_pool = AttentionPooling(num_channels)
        
        # Latent Projection via Linear layers
        # Input: [Batch, Channels] -> Output: [Batch, Latent_Dim]
        self.fc_mu = nn.Linear(num_channels, latent_dim)
        self.fc_logvar = nn.Linear(num_channels, latent_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc_mu.weight, mode='fan_in', nonlinearity='linear')
        nn.init.kaiming_normal_(self.fc_logvar.weight, mode='fan_in', nonlinearity='linear')
        
        # Initialize biases for stable KL starting point
        nn.init.constant_(self.fc_mu.bias, 0.0)
        nn.init.constant_(self.fc_logvar.bias, -2.0)  # σ² = exp(-2) ≈ 0.14, gives reasonable initial KL

    def forward(self, x):
        # x: [Batch, Input_Dim, Window]
        out = self.input_conv(x)
        out = self.tcn(out)
        
        # Attention Pooling: [Batch, Channels, Window] -> [Batch, Channels]
        out, attn_weights = self.attention_pool(out)
        
        mu = self.fc_mu(out)       # [Batch, Latent_Dim]
        logvar = self.fc_logvar(out)  # [Batch, Latent_Dim]
        return mu, logvar

class Decoder(nn.Module):
    """
    Conditional Decoder for cVAE.
    Broadcasts global z and concatenates with time-series condition c.
    Input: z [Batch, Latent_Dim], c [Batch, Window, Cond_Dim]
    Output: [Batch, Channels, Window]
    """
    def __init__(self, latent_dim, cond_dim, num_channels, output_dim, kernel_size, dropout, window_size, dilations):
        super(Decoder, self).__init__()
        
        self.window_size = window_size
        self.num_channels = num_channels
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim
        
        # Project broadcasted z + time-series c to hidden channels
        # Input: [Batch, Latent_Dim + Cond_Dim, Window]
        self.input_conv = nn.Conv1d(latent_dim + cond_dim, num_channels, kernel_size=1)
        
        layers = []
        num_levels = len(dilations)
        
        # TCN Residual Blocks to refine the sequence
        for i in range(num_levels):
            dilation_size = dilations[i]
            pad = (kernel_size - 1) * dilation_size // 2
            
            layers.append(
                TemporalBlock(
                    n_inputs=num_channels, 
                    n_outputs=num_channels, 
                    kernel_size=kernel_size, 
                    stride=1, 
                    dilation=dilation_size, 
                    padding=pad, 
                    dropout=dropout
                )
            )
            
        self.tcn = nn.Sequential(*layers)
        
        # Final projection to Target Dim
        self.final_conv = nn.Conv1d(num_channels, output_dim, kernel_size=1)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.final_conv.bias, 0)

    def forward(self, z, c):
        # z: [Batch, Latent_Dim] - Global latent vector
        # c: [Batch, Window, Cond_Dim] - Time-series condition features
        
        batch_size = z.size(0)
        
        # Broadcast z to window length: [Batch, Latent_Dim] -> [Batch, Latent_Dim, Window]
        z_broadcast = z.unsqueeze(-1).expand(-1, -1, self.window_size)
        
        # Permute c for Conv1d: [Batch, Window, Cond_Dim] -> [Batch, Cond_Dim, Window]
        c_permuted = c.permute(0, 2, 1)
        
        # Concatenate: [Batch, Latent_Dim + Cond_Dim, Window]
        z_cond = torch.cat([z_broadcast, c_permuted], dim=1)
        
        # Project to hidden channels
        out = self.input_conv(z_cond)  # [Batch, num_channels, Window]
        
        # Refine with TCN
        out = self.tcn(out)
        
        # Final projection (no activation - data is standardized, can be negative)
        out = self.final_conv(out)
        return out

class ImputationVAE(nn.Module):
    """
    Conditional VAE (cVAE) with Global Latent Vector for Time Series Imputation.
    
    Architecture:
        Input: x [Batch, Window, Target], c [Batch, Window, Aux], mask [Batch, Window, Target]
        Encoder: TCN -> Global Pool -> [Batch, Latent_Dim]
        Decoder: Broadcast z + concat c -> TCN -> [Batch, Window, Target]
    """
    def __init__(self, 
                 target_dim, 
                 aux_dim, 
                 window_size, 
                 latent_dim=256, 
                 hidden_dims=[256, 256, 256],
                 bottleneck_seq_len=None,  # Not used in global latent, kept for compatibility
                 encoder_layers=6,
                 decoder_layers=6): 
        super(ImputationVAE, self).__init__()
        
        self.target_dim = target_dim
        self.aux_dim = aux_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        
        # Config for TCN
        tcn_channels = hidden_dims[0] if hidden_dims else 256
        kernel_size = 3
        dropout = 0.2
        
        # Dynamically generate dilations based on layer counts
        enc_dilations = [2**i for i in range(encoder_layers)]
        dec_dilations = [2**i for i in range(decoder_layers)]
        
        # Input to encoder is [X, C, M]
        input_dim = target_dim + aux_dim + target_dim
        
        self.encoder = Encoder(
            input_dim=input_dim, 
            num_channels=tcn_channels, 
            kernel_size=kernel_size, 
            dropout=dropout, 
            latent_dim=latent_dim, 
            window_size=window_size,
            dilations=enc_dilations
        )
        
        # Decoder receives z + c (time-series condition)
        self.decoder = Decoder(
            latent_dim=latent_dim,
            cond_dim=aux_dim,  # Condition dimension = aux_dim
            num_channels=tcn_channels, 
            output_dim=target_dim, 
            kernel_size=kernel_size, 
            dropout=dropout, 
            window_size=window_size,
            dilations=dec_dilations
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c, mask):
        # Prepare Input
        # x, mask: [Batch, Window, Target]
        # c: [Batch, Window, Aux]
        
        # Concatenate: [Batch, Window, Target+Aux+Target]
        inputs = torch.cat([x, c, mask], dim=-1)
        
        # Permute for Conv1d: [Batch, Channels, Window]
        inputs = inputs.permute(0, 2, 1)
        
        # Encode to global latent
        mu, logvar = self.encoder(inputs)  # [Batch, Latent_Dim]
        
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        # Decode with condition (cVAE): z + time-series c
        recon = self.decoder(z, c)  # Decoder broadcasts z and concats with c
        
        # Permute back: [Batch, Window, Target]
        recon = recon.permute(0, 2, 1)
        
        return recon, mu, logvar