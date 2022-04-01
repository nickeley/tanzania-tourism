# Tanzania Tourism Prediction
The Tanzanian tourism sector plays a significant role in the Tanzanian economy, contributing about 17% to the country’s GDP and 25% of all foreign exchange revenues. The sector, which provides direct employment for more than 600,000 people and up to 2 million people indirectly, generated approximately $2.4 billion in 2018 according to government statistics. Tanzania received a record 1.1 million international visitor arrivals in 2014, mostly from Europe, the US and Africa.

Tanzania is the only country in the world which has allocated more than 25% of its total area for wildlife, national parks, and protected areas.There are 16 national parks in Tanzania, 28 game reserves, 44 game-controlled areas, two marine parks and one conservation area.

The objective of this project is to develop a machine learning model to predict what a tourist will spend when visiting Tanzania.The model can be used by different tour operators and the Tanzania Tourism Board to automatically help tourists across the world estimate their expenditure before visiting Tanzania.

</br>


The data for this project can be found on Zindi: **[Data](https://zindi.africa/competitions/tanzania-tourism-prediction)**.
</br>
Please put the data in the folder "data/original_zindi_data"

</br>


## Requirements:
- pyenv with Python: 3.9.4
</br>
</br>


## Environment to run the tensorflow part
Use the requirements file in this repo to create a new environment.
</br>
To run the tensorflow part, you have to install hdf5:

```BASH
 brew install hdf5
```
With the system setup like that, you can go and create your environment and install tensorflow

```BASH
pyenv local 3.9.4
python -m venv .venv
source .venv/bin/activate
export HDF5_DIR=/opt/homebrew/Cellar/hdf5/1.12.1

pip install --upgrade pip
pip install --no-binary=h5py h5py
pip install -r requirements.txt
```

If you do not want to run the neural network, you can just use the Makefile:
```BASH
make setup 
```


</br>

## Notebooks and other files in this repository
Jupyter Notebook with EDA on the data and Machine Learning Model can be found here: **[EDA Notebook](EDA-tanzania-tourism.ipynb)**.
</br>
An additional notebook with a Neural Network can be found here: **[Neural Network Notebook](tanzania-tourism-neural-network.ipynb)**.
</br>
The presentation can be found here: **[Presentation](Tanzania_Tourism_Presentation.pdf)**.
</br></br>

## Usage of scripts

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python train.py  
```

In order to test that predict works on a test set you created run:

```bash
python predict.py models/adaboost_model.sav data/X_test.csv data/y_test.csv
```
</br>

## Remarks
To reduce calculation time, a Randomized Search is used in the EDA Notebook and also in the Python Script. For this reason, the results of the Notebook and the Script can differ, so keep that in mind when trying things out.