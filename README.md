# Car vs Bike Classifier using Shallow Neural Network

This project implements a **binary image classifier** to distinguish between cars and bikes using a **shallow neural network** from scratch in **Numpy**.  

The model has **one hidden layer** with ReLU activation and a **sigmoid output** for binary classification.

---

## **Dataset**

- Custom dataset with two categories:
  - `Car/` → Images of cars  
  - `Bike/` → Images of bikes  
- Images are resized to **64x64 pixels** and converted from **BGR to RGB**.  
- Labels:  
  - 0 → Bike  
  - 1 → Car  

---

## **Data Preprocessing**

1. Resized images to 64×64 pixels.  
2. Converted BGR to RGB.  
3. Combined images into **numpy arrays** with assigned labels.  
4. Split dataset into **80% training** and **20% testing**.  
5. Normalized pixel values to `[0,1]`.  
6. Flattened images for input to the neural network.  
7. Saved dataset in **HDF5** format (`train_carbike.h5`).  

---

## **Model Architecture**

- **Shallow Neural Network**:
  - Input layer → Flattened image vectors
  - Hidden layer → 5 neurons, **ReLU activation**
  - Output layer → 1 neuron, **Sigmoid activation**
- **Loss Function:** Binary cross-entropy  
- **Optimizer:** Gradient Descent  

---

## **Training**

- Number of iterations: 1500  
- Learning rate: 0.01  
- Monitored cost during training every 100 iterations  

**Code to train:**

```python
W1, b1, W2, b2, costs = optimize_shallow(W1, b1, W2, b2, X_train_flatten, Y_train.T, num_iterations=1500, learning_rate=0.01)
