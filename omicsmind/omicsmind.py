import torch
import torch.nn as nn


class ModalityVAE(nn.Module):
    def __init__(self, in_dim, z_dim=64, h_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(h_dim, z_dim)
        self.fc_logvar = nn.Linear(h_dim, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, in_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z


class CrossOmicsTransformer(nn.Module):
    def __init__(
        self, z_dim=64, n_modalities=3, n_heads=4, n_layers=2, ff_dim=256, dropout=0.1
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=z_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos_emb = nn.Parameter(torch.randn(1, n_modalities, z_dim))

    def forward(self, Z):
        Z = Z + self.pos_emb[:, : Z.size(1)]
        return self.encoder(Z)


class OmicsMind(nn.Module):
    def __init__(self, in_dims, z_dim=64):
        super().__init__()
        self.mods = list(in_dims.keys())
        self.vaes = nn.ModuleDict(
            {m: ModalityVAE(in_dims[m], z_dim=z_dim) for m in self.mods}
        )
        self.trans = CrossOmicsTransformer(z_dim=z_dim, n_modalities=len(self.mods))

    def forward(self, batch_x, train_mask):
        Z_list, mus, logvars, refined_xhat = [], [], [], {}
        for m in self.mods:
            x_hat, mu, lv, z = self.vaes[m](batch_x[m])
            Z_list.append(z.unsqueeze(1))
            mus.append(mu)
            logvars.append(lv)

        Z = torch.cat(Z_list, dim=1)  # (B,M,z)

        mask_vec = torch.stack([train_mask[m] for m in self.mods], dim=1).unsqueeze(-1)
        Z_masked = Z * mask_vec

        Z_refined = self.trans(Z_masked)

        for i, m in enumerate(self.mods):
            refined_xhat[m] = self.vaes[m].decode(Z_refined[:, i, :])

        return refined_xhat, mus, logvars


def vae_kl(mu, logvar):
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()


def omicsmind_loss(
    batch_x,
    refined_xhat,
    mus,
    logvars,
    observed,
    train_mask,
    beta=1e-3,
    only_masked=False,
):
    rec = 0.0
    for m in batch_x:
        x_true = batch_x[m]
        x_pred = refined_xhat[m]
        obs_w = observed[m].float().unsqueeze(1)  # (B,1)

        if only_masked:
            m_w = (1 - train_mask[m]).float().unsqueeze(1)  # only masked modalities
            w = obs_w * m_w
        else:
            w = obs_w

        rec += (((x_true - x_pred) ** 2) * w).mean()

    kl = sum(vae_kl(mu, lv) for mu, lv in zip(mus, logvars))
    return rec + beta * kl, rec, kl
