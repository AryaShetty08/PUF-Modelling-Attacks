# simple logistic regression attack as base 
# should work on simple pufs 

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pypuf.io
import pypuf.simulation 
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

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

def trainLR(challenges, responses):
    # Train logistic regression model
    # lbfgs optimizer since data isn't huge? maybe have to change it
    train_challenges, val_challenges, train_responses, val_responses = train_test_split(challenges, responses, test_size=0.2, random_state=42)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(train_challenges, train_responses)
    
    # Calculate losses
    train_prob = clf.predict_proba(train_challenges)[:, 1]
    val_prob = clf.predict_proba(val_challenges)[:, 1]
    
    train_loss = log_loss(train_responses, train_prob)
    val_loss = log_loss(val_responses, val_prob)
    
    # Plot single point
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.bar(['Training Loss', 'Validation Loss'], [train_loss, val_loss])
    plt.title('Logistic Regression - Losses')
    plt.ylabel('Loss')
    plt.ylim(0,1)
    plt.subplot(1,2,2)
    accuracy = clf.score(val_challenges, val_responses)
    plt.bar(['Validation Accuracy'], [accuracy])
    plt.title('Logistic Regression - Accuracy')
    plt.ylabel('Accuracy')
    plt.ylim(0,1)
    plt.tight_layout()

    model_name = f"LR"

    plt.savefig(f'models/LR/2/{model_name.lower().replace(" ", "_")}_training_curves.png')
    print(f"Training curves saved to models/LR/2/{model_name.lower().replace(' ', '_')}_training_curves.png")

    return clf

def testLR(challenges, responses, model):
    # Test on new CRPs
    # make sure to apply responses before changing input
    pred_probs = model.predict_proba(challenges)[:, 1]
    
    confidence = np.abs(pred_probs - 0.5) * 2
    predictions = (pred_probs >= 0.5).astype(int)
    accuracy = np.mean(predictions == responses)

    high_conf_mask = confidence > 0.8
    high_conf_predictions = predictions[high_conf_mask]
    high_conf_responses = responses[high_conf_mask]
    
    if len(high_conf_predictions) > 0:
        high_conf_accuracy = np.mean(high_conf_predictions == high_conf_responses)
        print(f"Total predictions: {len(predictions)}")
        print(f"High confidence predictions: {np.sum(high_conf_mask)}")
        print(f"High confidence accuracy: {high_conf_accuracy:.4f}")

    print(f"Overall accuracy: {accuracy:.4f}")


    plt.figure(figsize=(10,5))
    plt.hist(confidence, bins=50, edgecolor='black')
    plt.title('Logistic Regression - Confidence Distribution')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Number of Predictions')
    plt.ylim(0,1000)
    # maybe change this line 
    plt.axvline(x=0.8, color='r', linestyle='--', label='High Confidence Threshold')
    plt.legend()
    
    model_name = f"LR"
    plt.savefig(f'models/LR/2/{model_name.lower().replace(" ", "_")}_confidence_distribution.png')
    print(f"Confidence curves saved to models/LR/2/{model_name.lower().replace(' ', '_')}_confidence_distribution.png")

    accuracy = model.score(challenges, responses)
    print(f"Attack Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    # parse input arguments 
    parser = argparse.ArgumentParser(description="PUF LR Training")
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
    else: # for none type
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
    model = trainLR(train_challenges, train_responses)

    # test acc of LR model 
    testLR(test_challenges, test_responses, model)

    with open("models/LR/2/params.txt", "w") as file:
        file.write(f"PUF type: {args.type}\n")
        file.write(f"n_bits: {n_bits}\n")
        file.write(f"num_crps: {num_crps}\n")

    joblib.dump(model, f"models/LR/2/lr_model.joblib")
    print("Model saved successfully!")
