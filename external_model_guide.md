# External Model Usage Guide

## Supported Model Formats

### PyTorch Models (.pt, .pth)
- **Full Models**: Complete model with architecture and weights
- **State Dictionaries**: Model weights only (requires matching architecture)

### Model Requirements
- Must be PyTorch models
- Should accept tensor inputs
- Must have `.eval()` method for evaluation mode

## Data Format Requirements

### Supported Formats
- **.npy**: Single numpy array
- **.npz**: Compressed numpy archive

### Data Structure Options

#### Option 1: Dictionary Format
```python
{
    'x_test': numpy_array,  # Input data
    'y_test': numpy_array   # Labels (optional)
}
```

#### Option 2: Direct Array
```python
numpy_array  # Direct input data
```

#### Common Key Names (Auto-detected)
- **Input data**: `x_test`, `x`, `data`, `X`
- **Labels**: `y_test`, `y`, `labels`, `Y`

## Input Shape Guidelines

### For CNN Models
- **4D**: `[batch, channels, height, width]` (e.g., `[100, 1, 28, 28]`)
- **3D**: `[batch, height, width]` (auto-adds channel dimension)

### For MLP Models  
- **2D**: `[batch, features]` (e.g., `[100, 784]`)
- **Higher dimensions**: Will be flattened automatically

## Example Usage

### 1. Prepare Your Model
```python
# Save your trained model
torch.save(model, 'my_model.pt')

# Or save state dict
torch.save(model.state_dict(), 'my_weights.pt')
```

### 2. Prepare Your Data
```python
# Option 1: Dictionary format
data = {
    'x_test': test_images,
    'y_test': test_labels
}
np.save('my_data.npy', data)

# Option 2: Direct array
np.save('my_images.npy', test_images)
```

### 3. Upload in VulneraX
1. Select "Upload External Model"
2. Choose your .pt/.pth file
3. Select "Upload External Data" 
4. Choose your .npy/.npz file
5. Select appropriate input format
6. Launch attack

## Troubleshooting

### Model Loading Issues
- **Error**: "State dict uploaded"
  - **Solution**: Use full model or built-in architectures

### Data Shape Issues  
- **Error**: Shape mismatch
  - **Solution**: Check input format selection (CNN vs MLP)

### Memory Issues
- **Error**: Out of memory
  - **Solution**: Use smaller batch sizes (app limits to 10 samples)

## Best Practices

1. **Test with small data first** (< 50 samples)
2. **Ensure model is in evaluation mode**
3. **Normalize input data** (0-1 range recommended)
4. **Match input shapes** to model expectations
5. **Use appropriate attack parameters** for your model type