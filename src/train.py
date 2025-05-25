import torch
from torch.optim import Adam
from pathlib import Path
from models.vae import VAE
from data_pipeline.dataloader import get_dataloader

def train(
    data_dir,
    output_dir,
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    latent_dim=128,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # Setup
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    dataloader = get_dataloader(data_dir, batch_size=batch_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            recon_batch, mu, log_var = model(data)
            loss = model.loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")

if __name__ == "__main__":
    train(
        data_dir="data/processed/hindi",
        output_dir="models/checkpoints"
    )