# VulneraX - Adversarial Attack Generator

A streamlined tool for testing neural network robustness against adversarial attacks.

## 🚀 Quick Installation

### Prerequisites
- Python 3.8+
- pip

### Install & Run
```bash
# Clone or download the project
cd VulneraX

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📋 App Overview

### Models Available
- **SimpleCNN**: Basic 2-layer CNN for 28x28 images
- **DeepCNN**: 3-layer CNN with dropout
- **SimpleMLP**: 4-layer fully connected network

### Test Datasets
- **small_dataset.npy**: 20 samples (quick testing)
- **mnist_like_data.npy**: 100 MNIST-like samples
- **synthetic_patterns.npy**: 30 pattern samples
- **flattened_data.npy**: 50 samples for MLP

### Attack Methods
- **FGSM**: Fast Gradient Sign Method (single-step)
- **PGD**: Projected Gradient Descent (multi-step)

## 🎯 How to Use

1. **Select Model**: Choose from SimpleCNN, DeepCNN, or SimpleMLP
2. **Pick Dataset**: Select test data (start with small_dataset.npy)
3. **Configure Attack**: 
   - Set epsilon (0.1-0.3 recommended)
   - For PGD: set iterations (20-40) and step size
4. **Launch Attack**: Click the button and view results
5. **Download**: Export adversarial examples as .npy file

## 📊 Results Display
- Success rate percentage
- Side-by-side image comparisons
- Prediction changes visualization
- Downloadable adversarial examples

## 🛡️ Security Features
- Input validation and sanitization
- Secure model loading (weights_only=True)
- Parameter bounds checking
- Safe data handling

## 📁 Project Structure
```
VulneraX/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── test_models/        # Pre-trained models
└── test_data/         # Test datasets
```

## 🔧 Troubleshooting

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Missing dependencies?**
```bash
pip install --upgrade -r requirements.txt
```

**Data loading errors?**
- Ensure test_data/ and test_models/ directories exist
- Try small_dataset.npy first

## 📝 License
MIT License - For research and educational purposes.