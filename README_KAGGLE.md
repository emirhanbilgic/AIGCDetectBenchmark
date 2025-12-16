# AIGC Detection Benchmark - Kaggle Version

This repository contains code for evaluating AI-generated image detection methods, with a focus on comparing different fake image generation techniques.

## Setup and Run on Kaggle

### 1. Create a New Kaggle Notebook

1. Go to [Kaggle](https://www.kaggle.com) and create a new notebook
2. Set the notebook to use GPU acceleration
3. Upload your `small_data` folder as a dataset or use the Kaggle dataset feature

### 2. Clone the Repository

```python
!git clone https://github.com/emirhanbilgic/AIGCDetectBenchmark.git
%cd AIGCDetectBenchmark
```

### 3. Install Dependencies

```python
!pip install -r requirements.txt
!pip install ftfy regex
```

### 4. Download UnivFD Weights

Since the weights are hosted on Baidu Pan, you'll need to download them manually and upload to Kaggle, or use a direct download method:

#### Option A: Upload weights to Kaggle dataset
1. Download UnivFD weights from: https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u
2. Create a Kaggle dataset with the weights
3. Mount the dataset in your notebook

#### Option B: Download directly (if accessible)
```python
# Note: Baidu Pan links may not work directly in Kaggle
# You'll likely need to upload the weights as a dataset
!mkdir -p weights/classifier
# Upload UnivFD.pth to weights/classifier/ manually
```

### 5. Prepare Your Data

Upload your `small_data` folder to Kaggle as a dataset or create it in the notebook:

```
small_data/
├── fake_ours/     # 100 PNG images
├── fake_semi-truths/  # 100 PNG images
└── real_images/   # ~100 JPG images
```

### 6. Run the Evaluation

```python
# Make sure you have the correct paths
!python eval_univfd_small_data.py \
  --model_path /path/to/your/UnivFD.pth \
  --small_data_root /path/to/small_data
```

### Example Complete Kaggle Notebook Code

```python
# Install dependencies
!pip install -r requirements.txt
!pip install ftfy regex

# Clone repo
!git clone https://github.com/emirhanbilgic/AIGCDetectBenchmark.git
%cd AIGCDetectBenchmark

# Create small_data directory structure
!mkdir -p small_data/fake_ours small_data/fake_semi-truths small_data/real_images

# Upload your data files to these directories
# (You'll need to manually upload files or mount a dataset)

# Create weights directory and upload UnivFD.pth
!mkdir -p weights/classifier
# Upload UnivFD.pth here

# Run evaluation
!python eval_univfd_small_data.py \
  --model_path weights/classifier/UnivFD.pth \
  --small_data_root small_data
```

## Expected Output

The script will evaluate UnivFD on:
- Real images vs. fake_ours
- Real images vs. fake_semi-truths

And determine which fake image set is harder to detect based on lower ROC-AUC scores.

## Troubleshooting

1. **Missing dependencies**: Make sure all packages from `requirements.txt` are installed
2. **CLIP model download**: The script will automatically download CLIP ViT-L/14 weights to `~/.cache/clip/`
3. **Memory issues**: Use GPU runtime and reduce batch size if needed
4. **Weights access**: Baidu Pan links may require VPN access from some regions

## Citation

If you use this code, please cite the original AIGC Detection Benchmark paper.
