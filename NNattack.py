# most intricate model
# hopefully should work on complicated PUFS

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import pypuf.io
import pypuf.simulation
import argparse
import matplotlib.pyplot as plt 

'''
Transforms raw challenge bits into features for PUF
The normal challenge bits represent the don't cross (1) and cross (-1)
which is not helpful for learning
The transformed bits are based on each PUF's architecture that show the effect each position has on the output

Input:
challenges - raw challenges array
PUF_type - transformation dependent

Output:
features - transformed array of challenges
'''
def input_map(challenges, PUF_type):

    # simplest PUF design with two paths switching, try using it for XOR as well
    if PUF_type == "Arbiter" or PUF_type == "XOR":
        # the challenges will start {-1, 1} format
        n = challenges.shape[1]
        features = np.zeros((challenges.shape[0], n), dtype=np.float32)
        
        # Transform challenges to reflect delay differences
        for i in range(n):
            # Calculate parity feature for position i
            features[:, i] = np.prod(challenges[:, i:], axis=1)
        
        print(features.shape)
        return features   

    # uses unique challenges fr each arbiter chain, tough to input map
    elif PUF_type == "LightweightSecure":
        n = challenges.shape[1]
        features = np.zeros((challenges.shape[0], n), dtype=np.float32)
        
        # For Lightweight Secure PUF, we need to transform challenges
        # by generating different challenge bits for each arbiter chain
        # First, we'll compute the base delay differences as in regular Arbiter PUF
        for i in range(n):
            features[:, i] = np.prod(challenges[:, i:], axis=1)
        
        # The key aspect of Lightweight Secure PUF is that it applies 
        # transformations to the input challenges before using them
        # This is often implemented with an XOR network that creates 
        # different effective challenges for each arbiter
        
        # We'll use a simple transformation here - you may need to adjust 
        # based on the specific implementation you're targeting
        transformed_features = features.copy()
        
        # Apply a simple mixing function to simulate the challenge transformation
        # In a real implementation, this would follow the specific circuit design
        for i in range(1, n):
            # Mix adjacent features to simulate challenge bit mixing
            transformed_features[:, i] = features[:, i] * features[:, i-1]
        
        return transformed_features

    # with no input mapping as base 
    return challenges


'''
    Main training function
    In this case uses logistic regression to train the PUF model 
    Effective for binary cases where you need to set output to either 0 or 1

    Input:
    challenges - transformed challenges after sending it through input_map
    responses - list that corresponds to the n-bit challenges, will be either 0 or 1

    Output:
    model - model that was trained with Neural Network 
    training plots - used matplotlib to show one iteration of training loss 
'''
def trainNN(challenges, responses):
    # Train NN Model
    num_crps, n_bits = challenges.shape
    print(n_bits, num_crps)

    # Split data into training and validation sets (90/10 split)
    split_idx = int(0.9 * num_crps)
    train_challenges = torch.tensor(challenges[:split_idx], dtype=torch.float32)
    train_responses = torch.tensor(responses[:split_idx], dtype=torch.float32)
    val_challenges = torch.tensor(challenges[split_idx:], dtype=torch.float32)
    val_responses = torch.tensor(responses[split_idx:], dtype=torch.float32)

    # Prepare dataloaders with appropriate batch size
    batch_size = 2048  # Increased batch size
    train_dataset = TensorDataset(train_challenges, train_responses)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TensorDataset(val_challenges, val_responses)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # NN model architecture
    class PUF_NN(nn.Module):
        def __init__(self, input_size):
            super(PUF_NN, self).__init__()
            
            self.model = nn.Sequential(
                # 4 Linear layers
                nn.Linear(input_size, 128),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(64),
                nn.Dropout(0.3),
                
                nn.Linear(64, 32),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(32),
                
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    # Initialize model, loss, and optimizer
    model = PUF_NN(n_bits)
    criterion = nn.BCELoss() # is this the right loss
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5) 

    # Training Loop
    print("Starting training...")
    num_epochs = 50
    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_responses in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_responses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_features.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch_features, batch_responses in val_loader:
                outputs = model(batch_features).squeeze()
                loss = criterion(outputs, batch_responses)
                val_loss += loss.item() * batch_features.size(0)
                
                val_preds.extend(outputs.round().cpu().numpy())
                val_true.extend(batch_responses.cpu().numpy())
    
        val_loss /= len(val_loader.dataset)
        val_accuracy = accuracy_score(val_true, val_preds)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
     
    with open("models/NN/3/params.txt", "w") as file:
        file.write(f"Epoch {num_epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}\n")

    name = f"NN"
    # Sent to plot function thought it was too clunky
    plot_train(name, train_losses, val_losses, val_accuracies)

    return model

'''
    Main testing function 
    Runs the saved model on new CRP set to see its effectiveness as model

    Input: 
    challenges - new transformed test challenge set 
    responses - new test responses for comparison to the model's predictions
    model - the trained model after running train script

    Output:
    Test accuracy
    Confidence plots - Shows the confidence of the model's predicitons

'''
def testNN(challenges, responses, model):

    test_challenges = torch.tensor(challenges, dtype=torch.float32)
    test_responses = torch.tensor(responses, dtype=torch.float32)
    
    criterion = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_challenges).squeeze()
        test_preds = test_outputs.round()
        test_accuracy = (test_preds == test_responses).float().mean().item()
        test_loss = criterion(test_outputs, test_responses).item()

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    with open("models/NN/3/params.txt", "a") as file:
        file.write(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n")

    # Calculate prediction confidence 
    confidence = abs(test_outputs - 0.5) * 2 
    high_conf_indices = confidence > 0.8
    if high_conf_indices.any():
        high_conf_accuracy = (test_preds[high_conf_indices] == test_responses[high_conf_indices]).float().mean().item()
        print(f"High confidence predictions: {high_conf_indices.sum().item()}/{len(test_outputs)}")
        print(f"High confidence accuracy: {high_conf_accuracy:.4f}")
    
        with open("models/NN/3/params.txt", "a") as file:
            file.write(f"High confidence predictions: {high_conf_indices.sum().item()}/{len(test_outputs)}\n")
            file.write(f"High confidence accuracy: {high_conf_accuracy:.4f}\n")

    plt.figure(figsize=(10,5))
    plt.hist(confidence, bins=50, edgecolor='black')
    plt.title('Neural Network - Confidence Distribution')
    plt.xlabel('Neural Network Confidence')
    plt.ylabel('Number of Predictions')
    plt.ylim(0,1000)
    # maybe change this line 
    plt.axvline(x=0.8, color='r', linestyle='--', label='High Confidence Threshold')
    plt.legend()
    model_name = f"NN"
    plt.savefig(f'models/NN/3/{model_name.lower().replace(" ", "_")}_confidence_distribution.png')
    print(f"Confidence curves saved to models/NN/3/{model_name.lower().replace(' ', '_')}_confidence_distribution.png")


'''
    Plotting function
    Not included in LR since I only thought it was too bulky for these train functions
    as they include the model

    Inputs:
    model_name
    train_losses
    val_losses
    val_accuracies

    Output:
    Two graphs for the training of the model 
'''
def plot_train(model_name, train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0, 1)  # Set y-axis limits
    plt.title(f'{model_name} - Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} - Validation Accuracy')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)  # Set y-axis limits
    
    plt.tight_layout()
    plt.savefig(f'models/NN/3/{model_name.lower().replace(" ", "_")}_training_curves.png')
    print(f"Training curves saved to models/NN/3/{model_name.lower().replace(' ', '_')}_training_curves.png")


'''
    Main input function 
    It takes in the input arguments when run and applies the correct PUF input mapping 
'''
if __name__ == "__main__":
    # reset txt file
    with open("models/NN/3/params.txt", "w") as file:
        file.write("Start\n")
    # parse input arguments 
    parser = argparse.ArgumentParser(description="PUF Neural Network Training")
    parser.add_argument("-t", "--type", default="None")
    args = parser.parse_args()

    # the main setup 
    n_bits = 64 # More bits = harder to attack
    num_crps = 100000  # More CRPs = better attack success???
    seed=1
    noisiness = 0.05
    k=4 # num of chains
    if args.type == "Arbiter":
        modelPUF = pypuf.simulation.ArbiterPUF(n=n_bits, seed=seed)
        modelCRP = pypuf.io.ChallengeResponseSet.from_simulation(modelPUF, N=num_crps, seed=2)
    elif args.type == "XOR":
        modelPUF = pypuf.simulation.XORArbiterPUF(n=n_bits, k=k, seed=seed)
        modelCRP = pypuf.io.ChallengeResponseSet.from_simulation(modelPUF, N=num_crps, seed=2)
    elif args.type == "Lightweight":
        modelPUF = pypuf.simulation.LightweightSecurePUF(n=n_bits, k=k, seed=seed)
        modelCRP = pypuf.io.ChallengeResponseSet.from_simulation(modelPUF, N=num_crps, seed=2)
    else: # for none type
        modelPUF = pypuf.simulation.ArbiterPUF(n=n_bits, seed=seed)
        modelCRP = pypuf.io.ChallengeResponseSet.from_simulation(modelPUF, N=num_crps, seed=2)   

    train_challenges = modelCRP.challenges   
    # Extract challenge and response data
    train_responses = modelCRP.responses.flatten()  # Convert from [[1], [1], [-1], ...] to [1, 1, -1, ...]
    # Convert responses (-1,1) → (0,1) for logistic regression
    train_responses = (train_responses + 1) // 2
    print("agaom", train_responses)

    test_challenges = pypuf.io.random_inputs(n=n_bits, N=1000, seed=42)
    test_responses = modelPUF.eval(test_challenges).flatten()
    test_responses = (test_responses + 1) // 2 

    # run specifc input mapping for required PUF 
    train_challenges = input_map(train_challenges, args.type)
    test_challenges = input_map(test_challenges, args.type)

    # run the training of NN model 
    model = trainNN(train_challenges, train_responses)

    # test acc of NN model 
    testNN(test_challenges, test_responses, model)

    # made txt file so we know the params of the model trained 
    with open("models/NN/3/params.txt", "a") as file:
        file.write(f"PUF type: {args.type}\n")
        file.write(f"n_bits: {n_bits}\n")
        file.write(f"num_crps: {num_crps}\n")
        file.write(f"k: {k}\n")

    # Save the model
    torch.save(model.state_dict(), f"models/NN/3/nn_model.pth")
    print("Model saved successfully!")