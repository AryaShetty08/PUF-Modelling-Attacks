# PUF Modelling Attacks
# Hardware security project

The project is compiled of three main scripts, LRattack, PerceptronAttack, and NNattack, and one main folder models which holds all the training results. 

If you want to run some of your own training and testing you can.

For all the scripts you can use None, Arbiter, XOR, or Lightweight PUF input mapping for training. The specific CRPs, noise, and other features are changed in the actual code since I didn't add a parameter. 

# Extra Parameters that are edited in script code:
  the main setup: 
    n_bits = 128 # More bits = harder to attack
    num_crps = 100000  # More CRPs = better attack success???
    seed=1
    noisiness = 0.05
    k=2 # num of chains

    (Specfic for Perceptron and NN)
    num_epochs = 50


In terms of testing after training, the script itself runs one test but I don't have a current testing script. Further implementation can be made however since I have added the model save feature, so the trained weights can be used again.

All the -t PUF input mapping types are:
- None
- Arbiter
- XOR
- Lightweight

# Linear Regression:
  Template: 
    python .\LRattack.py -t Arbiter
# Perceptron:
  Template: 
    python .\PerceptronAttack.py -t XOR      
# Neural Network:
  Template: 
    python .\NNattack.py -t None

After running these commands the scripts should train and test based on the parameters you set or changed in the code. For the acutal location of the results it will saved to the folder in models for the specifc attack you ran. In terms of the numbers, the file number will be in the code. As of right now I set them all to 9.             