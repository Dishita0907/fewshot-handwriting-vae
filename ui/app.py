from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
from PIL import Image
import io
from pathlib import Path
from src.models.vae import VAE

app = Flask(__name__)

def load_model(model_path):
    model = VAE()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    model_type = request.form.get('model_type', 'Hindi')
    model_path = f"models/checkpoints/final_model_{model_type.lower()}.pt"

    if not Path(model_path).exists():
        return jsonify({'error': f'Model for {model_type} not found. Please train the model first.'})

    model = load_model(model_path)
    
    with torch.no_grad():
        z = torch.randn(1, model.latent_dim)
        generated = model.decoder(z)
        
        # Convert tensor to image
        img_array = generated.squeeze().numpy()
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Convert image to bytes
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)
        
        return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)