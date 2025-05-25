import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from pathlib import Path
from src.models.vae import VAE

def load_model(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def main():
    st.title("Handwriting Font Generator")
    st.write("Generate handwriting fonts using VAE")

    # Sidebar for model selection
    model_type = st.sidebar.selectbox("Select Language", ["Hindi", "English"])
    model_path = f"models/checkpoints/final_model_{model_type.lower()}.pt"

    if not Path(model_path).exists():
        st.error(f"Model for {model_type} not found. Please train the model first.")
        return

    # Load model
    model = load_model(model_path)

    # Drawing interface
    st.write("Draw a character:")
    canvas_result = st.empty()  # Placeholder for drawing canvas

    if st.button("Generate"):
        if canvas_result is not None:
            # Process the drawn image
            # Convert to tensor and generate new sample
            with torch.no_grad():
                z = torch.randn(1, model.latent_dim)
                generated = model.decoder(z)
                
                # Convert tensor to image
                img_array = generated.squeeze().numpy()
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                
                # Display generated image
                st.image(img, caption="Generated Character", use_column_width=False)
        else:
            st.warning("Please draw a character first!")

if __name__ == "__main__":
    main()