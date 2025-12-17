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

Upload your `test_data` folder to Kaggle as a dataset. The script handles nested subfolders:

```
test_data/
├── fake_ours/            # Your fake images (with subfolders)
│   ├── celebahq_openjourney_ours/
│   ├── cityscapes_kandindsky_ours/
│   ├── openimages_stablediffusion_v4_ours/
│   └── sun_rgbd_kandinsky_ours/
├── fake_semi-truths/     # Semi-truths fake images (with subfolders)
│   ├── celebahq_openjourney/
│   ├── cityscapes_kandindsky/
│   ├── OpenImages_StableDiffusion_v4/
│   └── sun_rgbd_kandinsky/
└── real_images/          # Real images (with subfolders)
    ├── celebahq/
    ├── cityscapes_real/
    ├── openimages_real/
    └── sun_rgbd_real/
```

The script automatically finds all images in all subfolders.

### 6. Run the Evaluation

```python
# Make sure you have the correct paths
!python eval_univfd_small_data.py \
  --model_path /path/to/your/UnivFD.pth \
  --small_data_root /path/to/small_data
```

### Example Complete Kaggle Notebook Code

```python
# 🚨 CRITICAL: Fix numpy/sklearn compatibility BEFORE ANYTHING ELSE!
# This must be the FIRST cell you run, or you'll get import errors!
!pip uninstall -y numpy scikit-learn
!pip install numpy==1.23.5 scikit-learn==1.2.2

# Now install dependencies
!pip install -r requirements.txt
!pip install ftfy regex

# Clone repo
!git clone https://github.com/emirhanbilgic/AIGCDetectBenchmark.git
%cd AIGCDetectBenchmark

# Verify weights are available (already included in repo)
!ls -la weights/classifier/

# Copy your test_data from Kaggle dataset
# Replace 'your-dataset-name' with your actual dataset name containing test_data/
!cp /kaggle/input/your-dataset-name/test_data/* test_data/ -r 2>/dev/null || echo "Upload test_data as Kaggle dataset first"

# Run evaluation
!python eval_univfd_small_data.py \
  --model_path weights/classifier/UnivFD.pth \
  --small_data_root test_data

# Alternative: CrossEfficientViT evaluation (experimental)
# Download weights from: https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection
# Look for Google Drive links in README or Issues
# Upload to Kaggle as dataset, then use:
# !cp /kaggle/input/cross-efficient/pytorch/default/1/cross_efficient_vit.pth cross_efficient_vit.pth
# !python eval_crossefficientvit_small_data.py \
#   --model_path cross_efficient_vit.pth \
#   --small_data_root test_data
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

8. **AttributeError: 'Opt' object has no attribute 'isVal'**:
   - This is a bug in the evaluation script that has been fixed
   - Make sure you have the latest version of `eval_univfd_small_data.py`
   - The script needs `isVal = False` for evaluation mode

9. **_pickle.UnpicklingError: Weights only load failed** (PyTorch 2.6):
   - This is a PyTorch version compatibility issue that has been fixed
   - Make sure you have the latest version of `eval_univfd_small_data.py`
   - The script now uses `weights_only=False` for checkpoint loading

9. **_pickle.UnpicklingError: invalid load key, '<'**:
   - This error should no longer occur since weights are now included in the repository
   - If you encounter this, ensure you're using the latest version of the repository
   - The weights are already validated and properly formatted

## Citation

If you use this code, please cite the original AIGC Detection Benchmark paper.
