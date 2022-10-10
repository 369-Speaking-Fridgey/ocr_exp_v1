import mlflow
import os
"""
- 처음에 계속해서 CLI에 mlflow ui라고 입력을 해도 안떴었는데 이는 mlruns폴더에 ui로 보여 줄 수 있는 파일들이 저장이 되어 있기 떄문에
무조건 working directory로 이동을 해서 CLI 창에 입력을 해야 한다는 것이다.
- mlflow ui url: http://127.0.0.1:5000/#/experiments/{experiment_id}
"""
# (1) Create an experiment
try:
    experiment_id = mlflow.get_experiment_by_name("Speaking Fridgey").experiment_id

except:
    mlflow.create_experiment("Speaking Fridgey")
    experiment_id = mlflow.get_experiment_by_name("Speaking Fridgey").experiment_id
# (2) Fetch the Experiment made using the mlflow client
# experiment = client.get_experiment("2")
run = mlflow.start_run(experiment_id = experiment_id, run_name = "Text Detection")
with run:
    mlflow.log_param("ping", 0)
    mlflow.log_param("pong", 1)
# print(experiment.name)
# print(experiment.artifact_location) ## 처음에 experiment를 만들 때에 




## MLFLOW_TRACKING_URI 환경 변수를 설정해서 remote server uri를 연결해야 한다.
# 근데 tracking server의 HTTP 인증을 위한 사용자 이름과 비밀 번호도 설정해 주어야 한다.
# mlflow.set_tracking_uri('https://10.103.103.5:5000')
# 마땅히 따로 tracking을 위한 서버가 항상 열려 있는 상황이 아니기 때문에 


