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

**✅ Weights are now included in the repository! No manual download needed.**

The UnivFD weights (`UnivFD.pth`) are automatically available when you clone the repository. They were downloaded from the official [UniversalFakeDetect repository](https://github.com/WisconsinAIVision/UniversalFakeDetect).

#### Automatic Setup
```python
# Weights are already in the repository at weights/classifier/UnivFD.pth
# No additional download steps needed!
!ls weights/classifier/
# Should show: UnivFD.pth
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

# Verify weights are available (already included in repo)
!ls -la weights/classifier/

# Upload your data files to these directories
# (You'll need to manually upload files or mount a dataset)

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

1. **Numpy/Sklearn compatibility error** (ValueError: numpy.dtype size changed):
   ```python
   # Run this FIRST before anything else:
   !pip uninstall -y numpy scikit-learn
   !pip install numpy==1.23.5 scikit-learn==1.2.2
   ```

2. **Missing dependencies**: Make sure all packages from `requirements.txt` are installed
3. **CLIP model download**: The script will automatically download CLIP ViT-L/14 weights to `~/.cache/clip/`
4. **Memory issues**: Use GPU runtime and reduce batch size if needed
5. **Weights access**: Baidu Pan links may require VPN access from some regions

6. **Argument parsing error** ("unrecognized arguments: --small_data_root"):
   - Make sure you're running the correct script: `eval_univfd_small_data.py`
   - The script uses custom arguments, not the standard eval_all.py arguments
   - If you get TestOptions arguments in the error, the wrong script is running

7. **AttributeError: 'Opt' object has no attribute 'isTrain'**:
   - This is a bug in the evaluation script that has been fixed
   - Make sure you have the latest version of `eval_univfd_small_data.py`
   - The script needs `isTrain = False` for evaluation mode

8. **_pickle.UnpicklingError: Weights only load failed** (PyTorch 2.6):
   - This is a PyTorch version compatibility issue that has been fixed
   - Make sure you have the latest version of `eval_univfd_small_data.py`
   - The script now uses `weights_only=False` for checkpoint loading

9. **_pickle.UnpicklingError: invalid load key, '<'**:
   - This error should no longer occur since weights are now included in the repository
   - If you encounter this, ensure you're using the latest version of the repository
   - The weights are already validated and properly formatted

## Citation

If you use this code, please cite the original AIGC Detection Benchmark paper.
