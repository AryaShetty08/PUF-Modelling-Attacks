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
    clf - model that was trained with linear regression
    training plots - used matplotlib to show one iteration of training loss 
'''
def trainLR(challenges, responses):
    # lbfgs optimizer since data isn't huge? maybe have to change it
    train_challenges, val_challenges, train_responses, val_responses = train_test_split(challenges, responses, test_size=0.2, random_state=42)
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(train_challenges, train_responses)
    
    # Calculate losses for metrics 
    train_prob = clf.predict_proba(train_challenges)[:, 1]
    val_prob = clf.predict_proba(val_challenges)[:, 1]
    
    train_loss = log_loss(train_responses, train_prob)
    val_loss = log_loss(val_responses, val_prob)
    
    # Plot single point, since using sklearn doesn't use epochs like pytorch learning
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

    # Saves to file directory
    plt.savefig(f'models/LR/1/{model_name.lower().replace(" ", "_")}_training_curves.png')
    print(f"Training curves saved to models/LR/1/{model_name.lower().replace(' ', '_')}_training_curves.png")

    # return model 
    return clf

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
def testLR(challenges, responses, model):
    # gather predictions
    pred_probs = model.predict_proba(challenges)[:, 1]
    
    confidence = np.abs(pred_probs - 0.5) * 2
    predictions = (pred_probs >= 0.5).astype(int)
    accuracy = np.mean(predictions == responses)

    # change confidence level if needed 
    high_conf_mask = confidence > 0.8
    high_conf_predictions = predictions[high_conf_mask]
    high_conf_responses = responses[high_conf_mask]
    
    if len(high_conf_predictions) > 0:
        high_conf_accuracy = np.mean(high_conf_predictions == high_conf_responses)
        print(f"Total predictions: {len(predictions)}")
        print(f"High confidence predictions: {np.sum(high_conf_mask)}")
        print(f"High confidence accuracy: {high_conf_accuracy:.4f}")
            
        with open("models/LR/1/params.txt", "a") as file:
            file.write(f"Total predictions: {len(predictions)}\n")
            file.write(f"High confidence predictions: {np.sum(high_conf_mask)}\n")
            file.write(f"High confidence accuracy: {high_conf_accuracy:.4f}\n")
            file.write(f"Overall accuracy: {accuracy:.4f}\n")
            
    print(f"Overall accuracy: {accuracy:.4f}")

    # bar graph 
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
    plt.savefig(f'models/LR/1/{model_name.lower().replace(" ", "_")}_confidence_distribution.png')
    print(f"Confidence curves saved to models/LR/1/{model_name.lower().replace(' ', '_')}_confidence_distribution.png")

    accuracy = model.score(challenges, responses)
    print(f"Attack Accuracy: {accuracy:.4f}")

    with open("models/LR/1/params.txt", "a") as file:
        file.write(f"Attack Accuracy: {accuracy:.4f}\n")

'''
    Main input function 
    It takes in the input arguments when run and applies the correct PUF input mapping 
'''
if __name__ == "__main__":
    # reset txt file
    with open("models/LR/1/params.txt", "w") as file:
        file.write("Start\n")
    # parse input arguments 
    parser = argparse.ArgumentParser(description="PUF LR Training")
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
    # Convert responses (-1,1) → (0,1) for training
    train_responses = (train_responses + 1) // 2
    
    print("agaom", train_responses)

    test_challenges = pypuf.io.random_inputs(n=n_bits, N=1000, seed=42)
    test_responses = modelPUF.eval(test_challenges).flatten()
    test_responses = (test_responses + 1) // 2 

    # run specifc input mapping for required PUF 
    train_challenges = input_map(train_challenges, args.type)
    test_challenges = input_map(test_challenges, args.type)

    # run the training of LR model 
    model = trainLR(train_challenges, train_responses)

    # test acc of LR model 
    testLR(test_challenges, test_responses, model)

    # made txt file so we know the params of the model trained 
    with open("models/LR/1/params.txt", "a") as file:
        file.write(f"PUF type: {args.type}\n")
        file.write(f"n_bits: {n_bits}\n")
        file.write(f"num_crps: {num_crps}\n")

    joblib.dump(model, f"models/LR/1/lr_model.joblib")
    print("Model saved successfully!")
