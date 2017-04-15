# kaggle-cancer
https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening


This code is based on [this tensorflow tutorial]:https://www.tensorflow.org/tutorials/image_retraining. 

#How to run ? 

Step-1: Have your training data somewhere. This would be the path you will provide as one of the parameter while running "retrain.py" script. 

Step-2:Understand the useful flags present in the retrain.py script. 

Step-3: Start training the model by running following command : 
$python retrain.py --image_dir "/shared/kgcoe-research/mil/kaggle_data/cervical_cancer/train/"  <-- This is where I have stored the data. 

Step-4:(Optional) It is nice to see how your model is doing. If you are running locally then just type : 
$tensorboard --logdir /tmp/retrain_logs/ (Or whereever you are stoing the summary logs) 
Go to http://127.0.0.1:6006 to obserbe model's performance. 
 
If you are running remotely then, redo ssh with following flag: 
ssh -L 16006:127.0.0.1:6006 user_name@my_server
tensorboard --logdir /tmp/retrain_logs/
Go to http://127.0.0.1:16006 to see the model performance. 
