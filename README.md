# CAR_ML
CAR_ML is a machine learning framework (based on random forest algorithm with the PubChem fingerprints) for screen carcinogenic chemicals.

#Usage

Run ML_model_RF using a case:
python ML_MODEL_RF.py
Before executing the code, ensure that the file 'PubChem_1697_FS.csv' is located in the directory specified by os.chdir()

Run AD_RF_PubChem using a case:
python AD_RF_PubChem.py
Before executing this code, ensure you have already trained and saved the model, then load it here.

Note: Before running the code, make sure to modify the input data paths in the script to match the location of your own data files. 
