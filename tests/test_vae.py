import pytest
import torch
from src.models.vae import VAE

@pytest.fixture
def vae_model():
    return VAE(latent_dim=128)

def test_vae_encoder(vae_model):
    x = torch.randn(1, 1, 64, 64)
    mu, log_var = vae_model.encoder(x)
    assert mu.shape == (1, 128)
    assert log_var.shape == (1, 128)

def test_vae_decoder(vae_model):
    z = torch.randn(1, 128)
    output = vae_model.decoder(z)
    assert output.shape == (1, 1, 64, 64)

def test_vae_forward(vae_model):
    x = torch.randn(1, 1, 64, 64)
    recon_x, mu, log_var = vae_model(x)
    assert recon_x.shape == x.shape
    assert mu.shape == (1, 128)
    assert log_var.shape == (1, 128)

def test_vae_loss(vae_model):
    x = torch.randn(1, 1, 64, 64)
    recon_x = torch.sigmoid(torch.randn(1, 1, 64, 64))
    mu = torch.randn(1, 128)
    log_var = torch.randn(1, 128)
    loss = vae_model.loss_function(recon_x, x, mu, log_var)
    assert isinstance(loss.item(), float)