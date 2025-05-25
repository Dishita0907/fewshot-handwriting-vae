# Few-shot Handwriting Font Generator

A Variational Autoencoder (VAE) based font generator for English & Hindi characters.

## Features

- Generate handwriting-style fonts for English and Hindi characters
- Interactive web interface for drawing and generating characters
- Few-shot learning capabilities
- Support for both languages with separate models

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fewshot-handwriting-vae.git
cd fewshot-handwriting-vae
```

2. Create and activate conda environment:
```bash
conda env create -f environment.yml
conda activate handwriting-vae
```

## Usage

### Data Preprocessing

1. Place your raw images in the appropriate directories:
   - English: `data/raw/english/`
   - Hindi: `data/raw/hindi/`

2. Run preprocessing:
```bash
python src/data_pipeline/preprocess.py
```

### Training

Train the model:
```bash
python src/train.py
```

### Web Interface

Launch the Streamlit interface:
```bash
streamlit run ui/app.py
```

## Model Architecture

The VAE consists of:
- Encoder: Convolutional neural network that maps images to latent space
- Decoder: Deconvolutional network that generates images from latent vectors
- Latent space: 128-dimensional representation of character styles

## Testing

Run tests:
```bash
pytest tests/
```

## License

MIT License - see LICENSE file for details.