import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model', default= 'resnet34', help= 'The model name where you store the train.txt and validate.txt')
args = parser.parse_args()

file_path= os.getcwd()+'/data/results/'+args.model
train_file_path= file_path+'/train.log'
validate_file_path= file_path+'/val.log'

train_history= pd.read_table(train_file_path).as_matrix()
validate_history= pd.read_table(validate_file_path).as_matrix()

plt.plot(train_history[:,2], label= 'Train Top 1 Accuracy')
plt.plot(validate_history[:,2], label= 'Validate Top 1 Accuracy')
plt.legend()
plt.show()

plt.plot(train_history[:,1], label= 'Train Loss Over Epoch')
plt.plot(validate_history[:,1], label= 'Validate loss Over Epoch')
plt.legend()
plt.show()

plt.plot(train_history[:,3],label= 'Train Top 5 Accuracy')
plt.plot(validate_history[:,3], label= 'Validate Top 5 Accuracy')
plt.legend()
plt.show()



