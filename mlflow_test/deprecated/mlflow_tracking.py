import os
from random import random, randint
import mlflow
from mlflow import log_metric, log_param, log_artifacts

if __name__ == "__main__":
    ## log a parameter (key-value pair)
    log_param("Test Param1", randint(0, 100))
    
    ## log a metric: metrics can be updated througout the run
    log_metric("ping", random())
    log_metric("ping", random() + 100)
    
    ## log an artifact(=output file)
    os.makedirs("outputs", exist_ok = True)
    with open("outputs/test.txt", 'w') as fwrite:
        fwrite.write("Hello MLFLow")
    log_artifacts("outputs")
    
    
    
    