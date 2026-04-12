import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def summation_unit(inputs, weights, bias):
    return bias * weights[0] + np.dot(inputs, weights[1:])

def activation_step(y):
    return 1 if y >= 0 else 0

def activation_bipolar_step(y):
    if y > 0:
        return 1
    elif y == 0:
        return 0
    else:
        return -1

def activation_sigmoid(y):
    return 1 / (1 + np.exp(-np.clip(y, -500, 500)))

def activation_tanh(y):
    return np.tanh(y)

def activation_relu(y):
    return max(0, y)

def activation_leaky_relu(y, alpha=0.01):
    return y if y > 0 else alpha * y

def comparator_error(target, output):
    return target - output

def sum_squared_error(targets, outputs):
    return 0.5 * sum((t - o) ** 2 for t, o in zip(targets, outputs))


def train_perceptron(X, y, activation_fn, w0=10, w1=0.2, w2=-0.75,
                     lr=0.05, max_epochs=1000, conv_threshold=0.002):
    weights = [w0, w1, w2]  
    bias = 1
    epoch_errors = []

    for epoch in range(max_epochs):
        outputs = []

        for xi, ti in zip(X, y):
            y_net = summation_unit(xi, weights, bias)
            o = activation_fn(y_net)
            outputs.append(o)

            err = comparator_error(ti, o)
            weights[0] += lr * err * bias        
            weights[1] += lr * err * xi[0]       
            weights[2] += lr * err * xi[1]       

        epoch_out = [activation_fn(summation_unit(xi, weights, bias)) for xi in X]
        sse = sum_squared_error(y, epoch_out)
        epoch_errors.append(sse)

        if sse <= conv_threshold:
            print(f"  Converged at epoch {epoch + 1} | SSE = {sse:.6f}")
            return weights, epoch_errors, epoch + 1

    print(f"  Did not converge in {max_epochs} epochs | Final SSE = {epoch_errors[-1]:.6f}")
    return weights, epoch_errors, max_epochs

# and gate
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# xor gate
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("=" * 60)
print("A2: AND Gate - Step Activation")
print("=" * 60)
w_final, errors_a2, conv_epoch = train_perceptron(
    X_and, y_and, activation_step
)
print(f"  Final weights: W0={w_final[0]:.4f}, W1={w_final[1]:.4f}, W2={w_final[2]:.4f}")

plt.figure(figsize=(7, 4))
plt.plot(range(1, len(errors_a2) + 1), errors_a2, color='steelblue')
plt.axhline(0.002, color='red', linestyle='--', label='Conv. threshold (0.002)')
plt.xlabel("Epoch")
plt.ylabel("Sum Squared Error")
plt.title("A2: AND Gate – Step Activation | Error vs Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A2_AND_step_error.png", dpi=120)
plt.show()


activation_fns = {
    "Step":          activation_step,
    "Bipolar Step":  activation_bipolar_step,
    "Sigmoid":       activation_sigmoid,
    "ReLU":          activation_relu,
}

print("\n" + "=" * 60)
print("A3: AND Gate – Comparing Activation Functions")
print("=" * 60)

results_a3 = {}
plt.figure(figsize=(9, 5))

for name, fn in activation_fns.items():
    print(f"\n  [{name}]")
    y_train = np.array([-1 if v == 0 else 1 for v in y_and]) \
              if name == "Bipolar Step" else y_and
    _, errs, conv = train_perceptron(X_and, y_train, fn)
    results_a3[name] = conv
    plt.plot(range(1, len(errs) + 1), errs, label=f"{name} (conv@{conv})")

plt.axhline(0.002, color='black', linestyle='--', label='Threshold')
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A3: AND Gate – Activation Function Comparison")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A3_activation_comparison.png", dpi=120)
plt.show()

print("\n  Convergence Summary:")
for name, conv in results_a3.items():
    print(f"    {name:<15} : {conv} epochs")


print("\n" + "=" * 60)
print("A4: AND Gate – Varying Learning Rate")
print("=" * 60)

learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
conv_epochs_a4 = []

for lr in learning_rates:
    print(f"\n  lr = {lr}")
    _, _, conv = train_perceptron(X_and, y_and, activation_step, lr=lr)
    conv_epochs_a4.append(conv)

plt.figure(figsize=(7, 4))
plt.plot(learning_rates, conv_epochs_a4, marker='o', color='darkorange')
plt.xlabel("Learning Rate")
plt.ylabel("Epochs to Converge")
plt.title("A4: AND Gate – Learning Rate vs Epochs to Converge")
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A4_lr_vs_epochs.png", dpi=120)
plt.show()

# xor 
print("\n" + "=" * 60)
print("A5: XOR Gate – Step Activation (A2 repeated)")
print("=" * 60)

_, errors_xor_step, conv_xor = train_perceptron(
    X_xor, y_xor, activation_step
)
print("  (XOR is not linearly separable – perceptron cannot converge)")

print("\nA5: XOR Gate – Activation Function Comparison (A3 repeated)")
results_a5 = {}
plt.figure(figsize=(9, 5))
for name, fn in activation_fns.items():
    print(f"\n  [{name}]")
    y_train = np.array([-1 if v == 0 else 1 for v in y_xor]) \
              if name == "Bipolar Step" else y_xor
    _, errs, conv = train_perceptron(X_xor, y_train, fn)
    results_a5[name] = conv
    plt.plot(range(1, len(errs) + 1), errs, label=f"{name}")

plt.axhline(0.002, color='black', linestyle='--', label='Threshold')
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A5: XOR Gate – Activation Function Comparison")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A5_XOR_activation_comparison.png", dpi=120)
plt.show()

# Customer Transaction Classification

print("\n" + "=" * 60)
print("A6: Customer Transaction Classification")
print("=" * 60)

X_cust_raw = np.array([
    [20, 6, 2, 386],
    [16, 3, 6, 289],
    [27, 6, 2, 393],
    [19, 1, 2, 110],
    [24, 4, 2, 280],
    [22, 1, 5, 167],
    [15, 4, 2, 271],
    [18, 4, 2, 274],
    [21, 1, 4, 148],
    [16, 2, 4, 198],
])
y_cust = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Normalize features (min-max)
X_cust = (X_cust_raw - X_cust_raw.min(axis=0)) / \
         (X_cust_raw.max(axis=0) - X_cust_raw.min(axis=0) + 1e-8)

def train_perceptron_multi(X, y, activation_fn, lr=0.1, max_epochs=1000,
                           conv_threshold=0.002):
    n_features = X.shape[1]
    weights = np.random.uniform(-0.5, 0.5, n_features + 1)
    bias = 1
    epoch_errors = []

    for epoch in range(max_epochs):
        for xi, ti in zip(X, y):
            y_net = weights[0] * bias + np.dot(xi, weights[1:])
            o = activation_fn(y_net)
            err = ti - o
            weights[0] += lr * err * bias
            weights[1:] += lr * err * xi

        outputs = [activation_fn(weights[0] * bias + np.dot(xi, weights[1:]))
                   for xi in X]
        sse = sum_squared_error(y, outputs)
        epoch_errors.append(sse)

        if sse <= conv_threshold:
            print(f"  Converged at epoch {epoch + 1} | SSE = {sse:.6f}")
            return weights, epoch_errors, outputs

    print(f"  Did not converge | Final SSE = {epoch_errors[-1]:.6f}")
    outputs = [activation_fn(weights[0] * bias + np.dot(xi, weights[1:]))
               for xi in X]
    return weights, epoch_errors, outputs

np.random.seed(42)
w_cust, errs_cust, preds_cust = train_perceptron_multi(
    X_cust, y_cust, activation_sigmoid, lr=0.1
)

preds_binary = [1 if p >= 0.5 else 0 for p in preds_cust]
print(f"\n  Predictions : {preds_binary}")
print(f"  Targets     : {y_cust.tolist()}")
print(f"  Accuracy    : {accuracy_score(y_cust, preds_binary) * 100:.1f}%")

plt.figure(figsize=(7, 4))
plt.plot(errs_cust, color='green')
plt.axhline(0.002, color='red', linestyle='--', label='Threshold')
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A6: Customer Classification – Sigmoid Perceptron")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A6_customer_error.png", dpi=120)
plt.show()

# Pseudo-Inverse Comparison

print("\n" + "=" * 60)
print("A7: Pseudo-Inverse Weight Estimation")
print("=" * 60)

# Add bias column
X_aug = np.hstack([np.ones((X_cust.shape[0], 1)), X_cust])
# w = pinv(X) * y
w_pinv = np.linalg.pinv(X_aug) @ y_cust
preds_pinv = [1 if p >= 0.5 else 0 for p in X_aug @ w_pinv]
print(f"  Pseudo-inverse weights : {np.round(w_pinv, 4)}")
print(f"  Predictions            : {preds_pinv}")
print(f"  Targets                : {y_cust.tolist()}")
print(f"  Accuracy               : {accuracy_score(y_cust, preds_pinv) * 100:.1f}%")
print(f"  Perceptron Accuracy    : {accuracy_score(y_cust, preds_binary) * 100:.1f}%")

# A8: Back-Propagation Neural Network 

print("\n" + "=" * 60)
print("A8: Backpropagation – AND Gate (2-2-1 Network)")
print("=" * 60)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(o):
    return o * (1 - o)

def train_backprop(X, y, lr=0.05, n_hidden=2, max_epochs=1000,
                   conv_threshold=0.002, seed=42):
    np.random.seed(seed)
    n_in = X.shape[1]

    V = np.random.uniform(-0.05, 0.05, (n_in + 1, n_hidden))  
    W = np.random.uniform(-0.05, 0.05, (n_hidden + 1, 1))      

    epoch_errors = []

    for epoch in range(max_epochs):
        sse = 0

        for xi, ti in zip(X, y):
            xi = xi.reshape(-1, 1)    

            
            x_aug = np.vstack([[1], xi])           
            h_net = V.T @ x_aug                    
            h_out = sigmoid(h_net)            
            h_aug = np.vstack([[1], h_out])        
            o_net = W.T @ h_aug                 
            o_out = sigmoid(o_net)               

            err = ti - o_out.item()
            sse += 0.5 * err ** 2

            delta_k = o_out * (1 - o_out) * (ti - o_out)   

            delta_h = h_out * (1 - h_out) * (W[1:] * delta_k.item())  

            
            W += lr * (h_aug * delta_k.item())    
            V += lr * (x_aug * delta_h.T)         

        epoch_errors.append(sse)

        if sse <= conv_threshold:
            print(f"  Converged at epoch {epoch + 1} | SSE = {sse:.6f}")
            return V, W, epoch_errors

    print(f"  Did not converge | Final SSE = {epoch_errors[-1]:.6f}")
    return V, W, epoch_errors

V_and, W_and, errs_bp = train_backprop(X_and, y_and, lr=0.5, max_epochs=5000)

plt.figure(figsize=(7, 4))
plt.plot(errs_bp, color='purple')
plt.axhline(0.002, color='red', linestyle='--', label='Threshold')
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A8: Backpropagation AND Gate – Error vs Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A8_backprop_AND_error.png", dpi=120)
plt.show()

# Backpropagation – XOR Gate

print("\n" + "=" * 60)
print("A9: Backpropagation – XOR Gate (2-2-1 Network)")
print("=" * 60)

V_xor, W_xor, errs_bp_xor = train_backprop(X_xor, y_xor, lr=0.5, n_hidden=4, max_epochs=5000, seed=0)

plt.figure(figsize=(7, 4))
plt.plot(errs_bp_xor, color='teal')
plt.axhline(0.002, color='red', linestyle='--', label='Threshold')
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A9: Backpropagation XOR Gate – Error vs Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A9_backprop_XOR_error.png", dpi=120)
plt.show()

# A10: 2-Output Node Perceptron 
print("\n" + "=" * 60)
print("A10: AND Gate – 2 Output Nodes Backpropagation")
print("=" * 60)

# Remap targets
y_and_2out = np.array([[1, 0], [1, 0], [1, 0], [0, 1]], dtype=float)

def train_backprop_2out(X, Y, lr=0.05, n_hidden=2, max_epochs=1000,
                        conv_threshold=0.002, seed=42):
    np.random.seed(seed)
    n_in = X.shape[1]
    n_out = Y.shape[1]

    V = np.random.uniform(-0.05, 0.05, (n_in + 1, n_hidden))
    W = np.random.uniform(-0.05, 0.05, (n_hidden + 1, n_out))

    epoch_errors = []

    for epoch in range(max_epochs):
        sse = 0

        for xi, ti in zip(X, Y):
            xi = xi.reshape(-1, 1)
            ti = ti.reshape(-1, 1)

            # Forward
            x_aug = np.vstack([[1], xi])
            h_out = sigmoid(V.T @ x_aug)
            h_aug = np.vstack([[1], h_out])
            o_out = sigmoid(W.T @ h_aug)

            err = ti - o_out
            sse += 0.5 * np.sum(err ** 2)

            # Backward
            delta_k = o_out * (1 - o_out) * err          
            delta_h = h_out * (1 - h_out) * (W[1:] @ delta_k) 

            W += lr * (h_aug @ delta_k.T)
            V += lr * (x_aug @ delta_h.T)

        epoch_errors.append(sse)
        if sse <= conv_threshold:
            print(f"  Converged at epoch {epoch + 1} | SSE = {sse:.6f}")
            return V, W, epoch_errors

    print(f"  Did not converge | Final SSE = {epoch_errors[-1]:.6f}")
    return V, W, epoch_errors

V_2out, W_2out, errs_2out = train_backprop_2out(X_and, y_and_2out, lr=0.5, max_epochs=5000)

plt.figure(figsize=(7, 4))
plt.plot(errs_2out, color='brown')
plt.axhline(0.002, color='red', linestyle='--', label='Threshold')
plt.xlabel("Epoch")
plt.ylabel("SSE")
plt.title("A10: AND Gate 2-Output Nodes – Backprop Error")
plt.legend()
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A10_2output_AND_error.png", dpi=120)
plt.show()

# MLPClassifier – AND Gate & XOR Gate

print("\n" + "=" * 60)
print("A11: MLPClassifier – AND Gate & XOR Gate")
print("=" * 60)

for gate_name, Xg, yg in [("AND", X_and, y_and), ("XOR", X_xor, y_xor)]:
    mlp = MLPClassifier(
        hidden_layer_sizes=(4,),
        activation='logistic',
        learning_rate_init=0.05,
        max_iter=1000,
        random_state=42,
        tol=1e-4
    )
    mlp.fit(Xg, yg)
    preds = mlp.predict(Xg)
    print(f"\n  {gate_name} Gate:")
    print(f"    Predictions : {preds.tolist()}")
    print(f"    Targets     : {yg.tolist()}")
    print(f"    Accuracy    : {accuracy_score(yg, preds) * 100:.1f}%")
    print(f"    Iterations  : {mlp.n_iter_}")

# MLPClassifier – Customer Transaction Dataset

print("\n" + "=" * 60)
print("A12: MLPClassifier – Customer Transaction Dataset")
print("=" * 60)

mlp_cust = MLPClassifier(
    hidden_layer_sizes=(8, 4),
    activation='logistic',
    learning_rate_init=0.1,
    max_iter=2000,
    random_state=42
)
mlp_cust.fit(X_cust, y_cust)
preds_mlp_cust = mlp_cust.predict(X_cust)
print(f"  Predictions : {preds_mlp_cust.tolist()}")
print(f"  Targets     : {y_cust.tolist()}")
print(f"  Accuracy    : {accuracy_score(y_cust, preds_mlp_cust) * 100:.1f}%")
print(f"  Iterations  : {mlp_cust.n_iter_}")

print("\n" + "=" * 60)
print("All experiments complete. Plots saved to /mnt/user-data/outputs/")
print("=" * 60)
