import pandas as pd

read_path = 'C:/Course_Materials/Course/Dissertation_Project/data_in/'
save_path = 'C:/Course_Materials/Course/Dissertation_Project/data_out/'
file = pd.read_csv('path.txt', header=0, sep='\n')  # for different seasons & rate files
train_file = pd.read_csv('train_path.txt', header=0, sep='\n')  # for different seasons & rate files
test_file = pd.read_csv('test_path.txt', header=0, sep='\n')  # for different seasons & rate files
