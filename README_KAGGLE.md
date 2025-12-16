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

Download the weights directly in Kaggle using one of these methods:

#### Option A: Direct Download (Recommended)
```python
# Create weights directory
!mkdir -p weights/classifier

# Download UnivFD weights
# Note: Baidu Pan may require VPN access from some regions
# If direct download fails, use Option B
!wget -O weights/classifier/UnivFD.pth "https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u"

# Alternative: If wget doesn't work, try curl
# !curl -o weights/classifier/UnivFD.pth "https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u"
```

#### Option B: Upload as Kaggle Dataset (If direct download fails)
1. Download weights locally from: https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u
2. Create a Kaggle dataset named `univfd-weights` with the `UnivFD.pth` file
3. Mount the dataset in your notebook:
```python
# Copy weights from mounted dataset
!cp /kaggle/input/univfd-weights/UnivFD.pth weights/classifier/
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

## Complete Kaggle Notebook Example

Here's a complete, copy-paste ready Kaggle notebook:

```python
# Install system dependencies
!apt-get update && apt-get install -y wget curl

# Install Python dependencies
!pip install -r requirements.txt
!pip install ftfy regex tqdm

# Clone repository
!git clone https://github.com/emirhanbilgic/AIGCDetectBenchmark.git
%cd AIGCDetectBenchmark

# Create directories
!mkdir -p weights/classifier
!mkdir -p small_data/fake_ours small_data/fake_semi-truths small_data/real_images

# Download UnivFD weights
print("Downloading UnivFD weights...")
try:
    !wget -O weights/classifier/UnivFD.pth "https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u" --timeout=30
    print("✅ Weights downloaded successfully!")
except:
    print("❌ Direct download failed. Please:")
    print("1. Download weights locally from: https://pan.baidu.com/s/1dZz7suD-X5h54wCC9SyGBA?pwd=l30u")
    print("2. Upload as Kaggle dataset named 'univfd-weights'")
    print("3. Mount dataset and copy: !cp /kaggle/input/univfd-weights/UnivFD.pth weights/classifier/")
    print("4. Then run: !cp /kaggle/input/your-small-data/* small_data/ -r")

# Copy your small_data (upload as Kaggle dataset first)
# Replace 'your-dataset-name' with your actual dataset name
try:
    !cp /kaggle/input/your-small-data/* small_data/ -r
    print("✅ Data copied successfully!")
except:
    print("❌ Data copy failed. Make sure you uploaded small_data as a Kaggle dataset")
    print("Manual steps:")
    print("1. Create Kaggle dataset with your small_data folder")
    print("2. Mount it in this notebook")
    print("3. Copy files: !cp /kaggle/input/your-dataset-name/* small_data/ -r")

# Verify files
print("\n📁 Checking downloaded files:")
!ls -la weights/classifier/
!ls -la small_data/

# Run evaluation
print("\n🚀 Running UnivFD evaluation...")
!python eval_univfd_small_data.py \
  --model_path weights/classifier/UnivFD.pth \
  --small_data_root small_data

print("\n✅ Evaluation complete! Check results above.")
```

## Expected Output

The script will output something like:
```
Evaluating fake_ours vs real_images...
ACC: 0.85, AP: 0.87, AUC: 0.89
r_acc: 0.82, f_acc: 0.88

Evaluating fake_semi-truths vs real_images...
ACC: 0.78, AP: 0.81, AUC: 0.83
r_acc: 0.79, f_acc: 0.77

[verdict] Harder to detect (lower AUC): fake_semi-truths
```

## Troubleshooting

### Common Issues:

1. **Weights download fails**:
   ```python
   # Use this if direct download doesn't work
   !cp /kaggle/input/univfd-weights/UnivFD.pth weights/classifier/
   ```

2. **Data not found**:
   ```python
   # Check your dataset paths
   !ls /kaggle/input/
   !cp /kaggle/input/your-dataset-name/small_data/* small_data/ -r
   ```

3. **Memory issues**:
   ```python
   # Add to your notebook for GPU memory monitoring
   import torch
   torch.cuda.empty_cache()
   ```

4. **CLIP download**:
   - The script automatically downloads CLIP ViT-L/14 (~1.2GB)
   - This happens on first run and may take a few minutes

## Citation

If you use this code, please cite the original AIGC Detection Benchmark paper.
