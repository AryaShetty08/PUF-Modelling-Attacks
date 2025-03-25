import torch
import joblib
import numpy as np
import pypuf.io
import pypuf.simulation
import argparse

def testTorch(challenges, responses, model_path):
    model = torch.load(model_path)
    model.eval()
    
    test_challenges = torch.tensor(challenges, dtype=torch.float32)
    test_responses = torch.tensor(responses, dtype=torch.float32)
    
    with torch.no_grad():
        test_outputs = model(test_challenges).squeeze()
        test_preds = test_outputs.round()
        
        test_accuracy = (test_preds == test_responses).float().mean().item()
        
        print(f"PyTorch Model Test Accuracy: {test_accuracy:.4f}")
        return test_preds.numpy(), test_outputs.numpy()

def testLR(challenges, responses, model_path):
    model = joblib.load(model_path)
    
    pred_probs = model.predict_proba(challenges)[:, 1]
    predictions = (pred_probs >= 0.5).astype(int)
    
    accuracy = np.mean(predictions == responses)
    
    print(f"Scikit-learn Model Test Accuracy: {accuracy:.4f}")
    return predictions, pred_probs

def main():
    parser = argparse.ArgumentParser(description="PUF LR Training")
    parser.add_argument("-t", "--type", default="None")
    args = parser.parse_args()

    # PUF Configuration
    n_bits = 64
    num_test_crps = 1000
    
    if args.type == "Arbiter":
        modelPUF = pypuf.simulation.ArbiterPUF(n=n_bits, seed=1)
    elif args.type == "Interpose":
        modelPUF = pypuf.simulation.InterposePUF(n=64, k_up=8, k_down=8, seed=1, noisiness=.05)
    else: # for none type
        modelPUF = pypuf.simulation.ArbiterPUF(n=n_bits, seed=1)

    test_challenges = pypuf.io.random_inputs(n=n_bits, N=num_test_crps, seed=42)
 
    # Generate responses
    test_responses = modelPUF.eval(test_challenges).flatten()
    test_responses = (test_responses + 1) // 2  # Convert to (0,1)
    
    # Input mapping (use the same mapping as in training)
    from LRattack import input_map  # Import your input mapping function
    test_challenges = input_map(test_challenges, args.type)
        
    # Test LR Model
    print("\nTesting Scikit-learn Model:")
    sklearn_predictions, sklearn_probs = testLR(
        test_challenges, 
        test_responses, 
        f'models/LR/1/lr_model.joblib'
    )

    # Test Perceptron Model
    print("Testing PyTorch Model:")
    pytorch_predictions, pytorch_probs = testTorch(
        test_challenges, 
        test_responses, 
        f'models/Perceptron/1/perceptron_model.pth'
    )

    # Test NN Model
    print("Testing PyTorch Model:")
    pytorch_predictions, pytorch_probs = testTorch(
        test_challenges, 
        test_responses, 
        f'models/NN/1/nn_model.pth'
    )

if __name__ == "__main__":
    main()