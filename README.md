# PUF Modelling Attacks
# Hardware security project

Hello, this repository contains my hardware security development project. Here, I trained and tested different modeling attacks on multiple PUFs. You can use the scripts and add them to continue your research on PUF Security. 

The project is composed of three main scripts, LRattack, PerceptronAttack, and NNattack, and one main folder, models, which holds all the training results. 

The pufTrain.py, testModel.py, and notes.txt are not to be used they are just for testing as of now. 

If you want to run some of your training and testing, you can do so.

For all the scripts, you can use None (just Arbiter without input mapping), Arbiter, XOR, or Lightweight PUF input mapping for training. The specific CRPs, noise, and other features are changed in the actual code since I didn't add a parameter. 

# Extra Parameters that are edited in script code:
  The main setup: 
    n_bits = 128 # More bits = harder to attack
    num_crps = 100000  # More CRPs = better attack success???
    seed=1
    noisiness = 0.05
    k=2 # num of chains

    (Specific for Perceptron and NN)
    num_epochs = 50


In terms of testing after training, the script itself runs one test, but I don't have a current testing script. Further implementation can be made, however, since I have added the model save feature, the trained weights can be used again.

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

After running these commands, the scripts should train and test based on the parameters you set or changed in the code. The actual location of the results will be saved in the models folder for the specific attack you ran. In terms of the numbers, the file number will be in the code. As of right now, I have set them all to 9. 

The results you will get include:

- params.txt, tells you what parameters you trained on, and the losses and accuracies
- Confidence distribution graph, shows you a graph of confidence distributions plotted when testing
- model.pth, so you can use the weights in another testing scenario
- training loss graph, a graph for training and validation losses and accuracies 



# HELP Notes:
It should work but if you have trouble running make sure you have all the folders in your directory. 

Also make sure you have the required libraries installed like pypuf and the imports. Here is a pipfreeze you can reference.

asttokens==2.4.1
colorama==0.4.6
comm==0.2.2
contourpy==1.2.1
cycler==0.12.1
debugpy==1.8.5
decorator==5.1.1
executing==2.1.0
filelock==3.14.0
fonttools==4.52.4
fsspec==2024.5.0
intel-openmp==2021.4.0      
ipykernel==6.29.5
ipython==8.27.0
jedi==0.19.1
Jinja2==3.1.4
joblib==1.4.2
jupyter_client==8.6.2       
jupyter_core==5.7.2
kiwisolver==1.4.5
MarkupSafe==2.1.5
matplotlib==3.9.0
matplotlib-inline==0.1.7    
memory-profiler==0.61.0     
mkl==2021.4.0
mpmath==1.3.0
nest-asyncio==1.6.0
networkx==3.3
numpy==1.26.4
packaging==24.0
pandas==2.2.2
parso==0.8.4
pillow==10.3.0
platformdirs==4.3.3
prompt_toolkit==3.0.47      
psutil==6.0.0
pure_eval==0.2.3
Pygments==2.18.0
pyparsing==3.1.2
pypuf==2.2.0
python-dateutil==2.9.0.post0
pytz==2024.1
pywin32==306
pyzmq==26.2.0
scikit-learn==1.6.1
scipy==1.15.2
six==1.16.0
stack-data==0.6.3
sympy==1.12.1
tbb==2021.12.0
threadpoolctl==3.5.0
torch==2.2.2
torchaudio==2.2.2
torchvision==0.17.2
tornado==6.4.1
traitlets==5.14.3
typing_extensions==4.12.0
tzdata==2024.1
wcwidth==0.2.13