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
import os 

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

    if PUF_type == "Arbiter":
        # the challenges will start {-1, 1} format
        n = challenges.shape[1]
        features = np.zeros((challenges.shape[0], n), dtype=np.float32)
    
        # Transform challenges to reflect delay differences
        for i in range(n):
            # Calculate parity feature for position i
            features[:, i] = np.prod(challenges[:, i:], axis=1)
        
        print(features.shape)
        return features
    
    elif PUF_type == "Interpose":
        n = challenges.shape[1]
        # For Interpose PUF, we need more sophisticated features
        # We'll create features that capture both the upper and lower PUF

        # Basic parity features (like Arbiter PUF)
        arbiter_features = np.zeros((challenges.shape[0], n), dtype=np.float32)
        for i in range(n):
            arbiter_features[:, i] = np.prod(challenges[:, i:], axis=1)

        # Additional features for the interpose effect
        # Typically, the Interpose PUF uses the output of a smaller upper PUF
        # at a specific position in the lower PUF's chain

        # We'll model this by creating "interpose features" that capture
        # the interaction between different segments of the challenge

        # Determine interpose position (typically n/2 for n-bit challenges)
        interpose_pos = n // 2

        # Create features for upper PUF (first half of bits)
        upper_features = np.zeros((challenges.shape[0], interpose_pos), dtype=np.float32)
        for i in range(interpose_pos):
            upper_features[:, i] = np.prod(challenges[:, i:interpose_pos], axis=1)

        # Create interaction features that model how the upper PUF's output
        # might affect the lower PUF's stages
        interaction_features = np.zeros((challenges.shape[0], n-interpose_pos), dtype=np.float32)
        for i in range(n-interpose_pos):
            # Model interaction between upper PUF result and lower PUF stages
            # We approximate the upper PUF result with the parity of the first half
            upper_puf_approx = np.prod(challenges[:, :interpose_pos], axis=1)
            interaction_features[:, i] = upper_puf_approx * np.prod(challenges[:, interpose_pos+i:], axis=1)

        # Combine all features
        # We include original arbiter features, upper PUF features, and interaction features
        combined_features = np.hstack((arbiter_features, upper_features, interaction_features))

        return combined_features
     
    # with no input mapping
    return challenges

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

    # Key improvement 2: Enhanced NN model architecture
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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)  # Learning rate scheduler

    # Training Loop with early stopping
    print("Starting training...")
    num_epochs = 50
    train_losses = []
    val_losses = []
    val_accuracies = []
    # best_val_loss = float('inf')
    # patience = 5
    # patience_counter = 0
    # best_model_state = None

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
        
        # Update learning rate based on validation loss
        #scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        # if val_loss < best_val_loss:
        #    best_val_loss = val_loss
        #    best_model_state = model.state_dict().copy()
        #    patience_counter = 0
        # else:
        #    patience_counter += 1
            
        # if patience_counter >= patience:
        #    print(f"Early stopping at epoch {epoch+1}")
        #    break
    name = f"NN"

    plot_train(name, train_losses, val_losses, val_accuracies)

    return model

def testNN(challenges, responses, model):
    # Load best model for testing
    test_challenges = torch.tensor(challenges, dtype=torch.float32)
    test_responses = torch.tensor(responses, dtype=torch.float32)
    
    criterion = nn.BCELoss()
    # Create a separate test set

    #if best_model_state:
    #    model.load_state_dict(best_model_state)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_challenges).squeeze()
        test_preds = test_outputs.round()
        test_accuracy = (test_preds == test_responses).float().mean().item()
        test_loss = criterion(test_outputs, test_responses).item()

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Calculate prediction reliability statistics
    confidence = abs(test_outputs - 0.5) * 2  # Scale to 0-1 where 1 is most confident
    high_conf_indices = confidence > 0.8
    if high_conf_indices.any():
        high_conf_accuracy = (test_preds[high_conf_indices] == test_responses[high_conf_indices]).float().mean().item()
        print(f"High confidence predictions: {high_conf_indices.sum().item()}/{len(test_outputs)}")
        print(f"High confidence accuracy: {high_conf_accuracy:.4f}")
    
    plt.figure(figsize=(10,5))
    plt.hist(confidence, bins=50, edgecolor='black')
    plt.title('Perceptron - Confidence Distribution')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Number of Predictions')
    plt.ylim(0,1000)
    # maybe change this line 
    plt.axvline(x=0.8, color='r', linestyle='--', label='High Confidence Threshold')
    plt.legend()
    model_name = f"NN"
    plt.savefig(f'models/NN/2/{model_name.lower().replace(" ", "_")}_confidence_distribution.png')
    print(f"Confidence curves saved to models/NN/2/{model_name.lower().replace(' ', '_')}_confidence_distribution.png")

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
    plt.savefig(f'models/NN/2/{model_name.lower().replace(" ", "_")}_training_curves.png')
    print(f"Training curves saved to models/NN/2/{model_name.lower().replace(' ', '_')}_training_curves.png")


if __name__ == "__main__":
      # parse input arguments 
    parser = argparse.ArgumentParser(description="PUF Neural Network Training")
    parser.add_argument("-t", "--type", default="None")
    args = parser.parse_args()

    # the main setup 
    # i guess have args to change what type of PUF, will do it later
    # Generate a dataset of Challenge-Response Pairs (CRPs)
    n_bits = 64
    num_crps = 100000  # More CRPs = better attack success

    if args.type == "Arbiter":
        modelPUF = pypuf.simulation.ArbiterPUF(n=n_bits, seed=1)
        modelCRP = pypuf.io.ChallengeResponseSet.from_simulation(modelPUF, N=num_crps, seed=2)
    elif args.type == "Interpose":
        modelPUF = pypuf.simulation.InterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=.05)
        modelCRP = pypuf.io.ChallengeResponseSet.from_simulation(modelPUF, N=num_crps, seed=2)
    else:
        modelPUF = pypuf.simulation.ArbiterPUF(n=n_bits, seed=1)
        modelCRP = pypuf.io.ChallengeResponseSet.from_simulation(modelPUF, N=num_crps, seed=2)

    train_challenges = modelCRP.challenges  # Shape: (50000, 64)    
    # Extract challenge and response data
    train_responses = modelCRP.responses.flatten()  # Convert from [[1], [1], [-1], ...] to [1, 1, -1, ...]
    # Convert responses (-1,1) → (0,1) for logistic regression
    train_responses = (train_responses + 1) // 2
    print("agaom", train_responses)

    test_challenges = pypuf.io.random_inputs(n=n_bits, N=1000, seed=42)
    test_responses = modelPUF.eval(test_challenges).flatten()
    test_responses = (test_responses + 1) // 2  # Convert to (0,1)

    # run specifc input mapping for required PUF 
    train_challenges = input_map(train_challenges, args.type)
    test_challenges = input_map(test_challenges, args.type)

    # run the training of LR model 
    model = trainNN(train_challenges, train_responses)

    # test acc of LR model 
    testNN(test_challenges, test_responses, model)

    with open("models/NN/2/params.txt", "w") as file:
        file.write(f"PUF type: {args.type}\n")
        file.write(f"n_bits: {n_bits}\n")
        file.write(f"num_crps: {num_crps}\n")

    # change name for each type 
    # Save the model
    torch.save(model.state_dict(), f"models/NN/2/nn_model.pth")
    print("Model saved successfully!")

    # MAKE GRAPHS 