# Adversarial Attack Types in VulneraX

## Overview
VulneraX generates **adversarial examples** - inputs that are intentionally designed to fool machine learning models by adding imperceptible perturbations to legitimate inputs.

## Attack Types Available

### 1. FGSM (Fast Gradient Sign Method)

#### What it does:
- **Single-step attack** that adds noise in one direction
- Finds the direction that maximally increases the model's prediction error
- Adds small perturbation in that direction

#### How it works:
```python
# 1. Calculate gradient of loss with respect to input
gradient = ∇x Loss(model(x), true_label)

# 2. Take sign of gradient (direction only)
sign_gradient = sign(gradient)

# 3. Add epsilon-scaled perturbation
adversarial_x = x + epsilon * sign_gradient
```

#### Characteristics:
- **Speed**: Very fast (single step)
- **Effectiveness**: Moderate success rate
- **Visibility**: Perturbations may be slightly visible
- **Use case**: Quick testing, baseline attacks

#### Parameters:
- **Epsilon (ε)**: Perturbation strength (0.0-1.0)

### 2. PGD (Projected Gradient Descent)

#### What it does:
- **Multi-step iterative attack** that refines perturbations
- Takes many small steps toward fooling the model
- Projects back to allowed perturbation space after each step

#### How it works:
```python
# Start with original input
x_adv = x

# Iterate multiple times
for i in range(num_iterations):
    # 1. Calculate gradient
    gradient = ∇x Loss(model(x_adv), target)
    
    # 2. Take small step
    x_adv = x_adv + alpha * sign(gradient)
    
    # 3. Project back to epsilon ball
    perturbation = clip(x_adv - x, -epsilon, epsilon)
    x_adv = clip(x + perturbation, 0, 1)
```

#### Characteristics:
- **Speed**: Slower (multiple iterations)
- **Effectiveness**: Higher success rate than FGSM
- **Visibility**: Better control over perturbation visibility
- **Use case**: More thorough security testing

#### Parameters:
- **Epsilon (ε)**: Maximum perturbation strength
- **Alpha (α)**: Step size per iteration
- **Iterations**: Number of refinement steps

## Attack Categories

### By Target:
- **Untargeted Attacks**: Make model predict ANY wrong class
- **Targeted Attacks**: Make model predict SPECIFIC wrong class
- *VulneraX uses untargeted attacks*

### By Knowledge:
- **White-box Attacks**: Full access to model (gradients, weights)
- **Black-box Attacks**: Only access to model outputs
- *VulneraX performs white-box attacks*

### By Perturbation:
- **L∞ Attacks**: Limit maximum change per pixel
- **L2 Attacks**: Limit total perturbation energy
- *VulneraX uses L∞ norm (epsilon bounds)*

## What These Attacks Test

### 1. Model Robustness
- How easily can the model be fooled?
- What's the minimum perturbation needed?
- Which inputs are most vulnerable?

### 2. Security Vulnerabilities
- Can attackers exploit the model in real-world scenarios?
- How visible would malicious inputs be?
- What's the attack success rate?

### 3. Generalization Issues
- Does the model rely on spurious features?
- Is it overfitting to training data patterns?
- How stable are the learned representations?

## Real-World Attack Scenarios

### Image Classification
- **Medical**: Fool diagnostic systems
- **Security**: Bypass facial recognition
- **Autonomous vehicles**: Misclassify traffic signs

### Example Attack Flow:
```
Original Image: Stop Sign → Model Predicts: "Stop Sign" ✓
    ↓ (Add tiny noise)
Adversarial Image: Stop Sign* → Model Predicts: "Speed Limit" ✗
```

## Attack Success Metrics

### Success Rate
- **High (>80%)**: Model is vulnerable
- **Medium (40-80%)**: Moderate robustness
- **Low (<40%)**: Good robustness

### Perturbation Visibility
- **ε < 0.1**: Usually imperceptible
- **ε = 0.1-0.3**: Slightly noticeable
- **ε > 0.3**: Clearly visible

## Defense Implications

### What Attacks Reveal:
1. **Gradient-based vulnerabilities**
2. **Decision boundary instabilities**
3. **Feature representation weaknesses**
4. **Training data biases**

### Common Defenses:
- **Adversarial Training**: Train with adversarial examples
- **Gradient Masking**: Hide gradients from attackers
- **Input Preprocessing**: Denoise inputs before prediction
- **Certified Defenses**: Provable robustness guarantees

## VulneraX Attack Pipeline

```
1. Load Model → 2. Load Data → 3. Select Attack Type
         ↓
4. Set Parameters → 5. Generate Adversarial Examples
         ↓
6. Evaluate Success → 7. Visualize Results → 8. Download
```

## Key Takeaways

- **FGSM**: Fast but basic attack for initial testing
- **PGD**: More sophisticated attack for thorough evaluation
- **Both test different aspects** of model robustness
- **Results indicate security vulnerabilities** in AI systems
- **Essential for responsible AI deployment**

These attacks help identify weaknesses before models are deployed in critical applications!

Both attacks create adversarial examples - inputs that look normal to humans but fool AI models into making wrong predictions. This helps identify security weaknesses before deploying models in critical applications.
