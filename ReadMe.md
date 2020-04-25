Documentation
=============
**General Description:**

This project mainly focuses on optimizing the operating strategy of a thermal plant, in order to maximize the power generation in each day.

Project mainly includes four parts:
1. data pre-processing part
2. feature extraction part
3. classification part
4. strategy optimization part


**Caution**: Make sure data has been correctly copied into the input data folder (data_in) when running the project.

Environment
-------------
**Interpreter:** python 3.5, MATLAB 2020a

package             | version       
------------------- | --------------
**numpy**|1.18.2
**matplotlib**|3.0.3
**pandas**|0.25.3
**tensorflow**|2.1.0
**sklearn**|0.0
**seaborn**|0.9.1


Code File
-------------
#### Pre-processing module
**Tasks:**
1. detect outliers via Isolation Forest and refill outliers via Just-in-time learning
2. partition raw data based on dates

Code file              |Description                       |Input data path                                                    |Output data path 
-----------------------|----------------------------------|-------------------------------------------------------------------|-----------------
outlier_utils.py       |detect and refill outliers        |match&classify/                                                    |outlier_processed/
preprocessing_main.py  |main function for pre-processing  |None                                                               |None
plot.py                |plot relevant figures             |outlier_processed/ (correlation heatmap) <br> merge.csv (raw data) |Figure/correlation(correlation heatmap)<br>Figure/Initial(raw data)
segmentation_utils.py  |data partition                    |merge.csv                                                          |match&classify/
set_path.py            |set path for module               |None                                                               |None

#### Feature Extraction module
**Tasks:**
1. extract linear principal components via PCA
2. extract non-linear principal components via autoencoder

Code file             |Description                                 |Input data path                                                   |Output data path 
----------------------|--------------------------------------------|------------------------------------------------------------------|-----------------
autoencoder_utils.py  |extract non-linear feature via autoencoder  |Autoencoder/train_data/                                           |AutoEncoder/data/（feature）<br> AutoEncoder/model/（model）
feature_main.py       |main function for feature extraction        |None                                                              |None
PCA_utils.py          |extract linear feature vai PCA              |outlier_processed/                                                |PCA/
plot_utils.py         |plot relevanr figures                       |PCA/ (PCA feature) <br> AutoEncoder/Example (autoencoder feature) |Figure/PCA/ (PCA feature) <br> Figure/AutoEncoder/ (autoencoder feature)
set_path.py           |set path for module

#### Classification module
**Tasks:**
1. discretize operating & weather condition through K-Means and DBSCAN clustering
2. classify unlabeled data via random forest

Code file              |Description                      |Input data path                                                      |Output data path 
-----------------------|---------------------------------|---------------------------------------------------------------------|-----------------
classifier_main.py     |main function for classification |None                                                                 |None
dbscan_utils.py        |DBSCAN clustering algorithm      |AutoEncoder/assemble.csv <br> PCA/homo(hetero)_standard/assemble.csv |classifier/DBScan_result.csv
kmeans_utils.py        |K-Means clustering               |AutoEncoder/assemble.csv <br> PCA/homo(hetero)_standard/assemble.csv |Kmeans/result(X).csv
plot_utils.py          |plot relevant figures            |KMeans/season/ar/                                                    |Figure/KMeans/
RandomForest.py        |random forest classifier         |classifier/RandomForest/                                             |classifier/RandomForest/result/
segmentation_utils.py  |data partition                   |classifier/Kmeans/result.csv                                         |classifier/KMeans
set_path.py            |set path for module              |None                                                                 |None
  
#### Optimization module
**Tasks:**
1. create markov decision process environment
2. solve MDP and optimize operating strategy

Code file          |Description                                       |Input data path  |Output data path 
-------------------|--------------------------------------------------|-----------------|-----------------
LoadData.m         |load training data for RL agent                   |train_data/      |None
myResetFunction.m  |define reset state for RL agent                   |None             |None
myStepFunction.m   |compute observation and reward after each action  |None             |None
Optimization.m     |Deep Q-Learning module for optimization           |None             |Strategy.xlsx
Update.m           |acquire training data from data set               |None             |None

Text File
-------------

Text File         |Description
------------------|-----------
attr_col1.txt     |column index for each subcategories in raw data
attr_col2.txt     |column index for each main categories in raw data
path.txt          |path for partitioned data
predict_path.txt  |path for predict data
test_path.txt     |path for test data
train_path.txt    |path for training data