"""pLDDT-prediction training script with bio-transformers & MLR"""
import os
import random
import os
from  typing import List
import neptune.new as neptune
import numpy as np
from biotransformers import BioTransformers
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import neptune.new.integrations.sklearn as npt_utils

from utils import create_dataset


if __name__ == "__main__":
    #init neptune logger
    run = neptune.init(project="sophiedalentour/pLDDT-prediction",
                    name= 'regression_model',
                    tags = ['RandomForestRegressor', 'regression']
                )

    
    # set the seed
    SEED = 42
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
  

    # embedding and convolution parameters
    BIOTF_MODEL = "esm1_t6_43M_UR50S"
    BIOTF_POOLMODE = "mean"
    BIOTF_BS = 4

    # training parameters
    TRAIN_SET = "datasets/train_min.csv"
    TEST_SET = "datasets/test_min.csv"

    
    # create train dataset
    sequences_train, labels_train = create_dataset(data_path=TRAIN_SET)
    
    # create test dataset
    sequences_test, labels_test = create_dataset(data_path=TEST_SET)


    # sequences embeddings with biotransformers
    bio_trans = BioTransformers(backend=BIOTF_MODEL)

    sequences_train_embeddings = bio_trans.compute_embeddings(
        sequences_train, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ] 
   
    sequences_test_embeddings = bio_trans.compute_embeddings(
        sequences_test, pool_mode=(BIOTF_POOLMODE,), batch_size=BIOTF_BS
    )[
        BIOTF_POOLMODE
    ] 


    # creating an object of Regression class
    LR = RandomForestRegressor()
    LR.fit(sequences_train_embeddings, labels_train)
    Y_Pred = LR.predict(sequences_test_embeddings)

    #Calculating the Accuracy
    test_set_rmse = (np.sqrt(mean_squared_error(labels_test, Y_Pred)))
    test_set_r2 = r2_score(labels_test, Y_Pred)

    # save parameters in neptune

    run["hyper-parameters"] = {
        "encoding_mode": "bio-transformers",
        "seed": SEED,
        "train_set": TRAIN_SET,
        "test_set": TEST_SET,
    }
    run['estimator/parameters'] = npt_utils.get_estimator_params(LR)
    run['estimator/pickled-model'] = npt_utils.get_pickled_model(LR)
    run['rfr_summary'] = npt_utils.create_regressor_summary(LR, 
        sequences_train_embeddings, 
        sequences_test_embeddings, 
        labels_train,
        labels_test
    )


    run.stop()
