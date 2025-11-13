## Setup & Installation Commands:

### Update System Packages
```
sudo apt update 
sudo apt upgrade -y
sudo apt install -y software-properties-common curl git build-essential
sudo apt update
```

### Verify Python Installation
```
python3 --version
```

### Note:Tested on Python 3.10.12 and ubuntu 22.04

### Install/Upgrade PIP
```
sudo apt install -y python3-pip
```

### Create New Directory
```
mkdir ~/disease_treatment
```

### Create CSV File Folder
```
mkdir ~/disease_treatment/csv_files
```

### Check CSV File Folder Creation
```
ls ~/disease_treatment
```

### Go to CSV File Directory Folder
```
cd ~/disease_treatment/csv_files
```

### Download both CSV Files from Disease_Treatment_V2_Repository
```
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Treatment_API_V2/refs/heads/testing/csv_files/disease_treatments_directory.csv
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Treatment_API_V2/refs/heads/testing/csv_files/radiology_pathology_test_terms.csv
```

### Verify that both files are downloaded
```
ls
```

### Go back to Main Directory
```
cd ~/disease_treatment
```

### Download app.py file from Repository
```
wget https://raw.githubusercontent.com/dhavalrohit/Disease_Treatment_API_V2/refs/heads/testing/app.py
```

### Verify download
```
ls
```

### Go Back to Main Directory
```
cd ~/disease_treatment
```

### Create Virtual Environement(Using any Custom Name here 'disease_treatment_venv' is used)
```
python3 -m venv disease_treatment_venv
```

### Activate Virtual Environement
```
source disease_treatment_venv/bin/activate
```

### Update Python/PIP
```
sudo apt install -y python3-pip
sudo apt update
sudo apt upgrade -y
```

### Install required dependencies
```
pip install gunicorn 
pip install flask pyngrok pandas flask-ngrok 
pip install spacy==3.8.7 numpy==2.2.6
```
### Install required NLP models
```
pip install "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
```
```
pip install "en-core-med7-lg @ https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl"
```
### Install other required dependencies
```
pip install --upgrade h5py
pip install scikit-learn
pip install joblib
```

## Running the API
```
cd ~/disease_treatment
```
```
gunicorn --bind 0.0.0.0:5000 app:app --log-level info --capture-output --enable-stdio-inheritance
```

### Command to Check if Gunicorn Process is active(API is running)
```
sudo lsof -i:5000
```
### Similar Output you should see
```
COMMAND    PID   USER   FD   TYPE DEVICE SIZE/OFF NODE NAME
gunicorn 28083 ubuntu    5u  IPv4 242028      0t0  TCP *:5000 (LISTEN)
gunicorn 28084 ubuntu    5u  IPv4 242028      0t0  TCP *:5000 (LISTEN)
```

### Test both Endpoints
http://<your_public_IP>:5000/extract
http://<your_public_IP>:5000/submit_feedback

### To Check Submit_feedback endpoint(List Last 10 rows in CSV File)
```
cd ~/disease_treatment/csv_files
tail -n 20 training_data.csv
```

### Command to Stop the API(Gunicorn proccess) if Actively Running in Background
```
sudo pkill gunicorn
```

## Troubleshooting
### If you encounter the following error:
#### import NumpyOps File "thinc\backends\numpy_ops.pyx", line 1, in init thinc.backends.numpy_ops ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
### Run the following commands to fix it:
```
pip uninstall numpy thinc spacy -y
pip install numpy==2.2.6
pip install spacy==3.8.7
```

## Steps to Host Your API(Using tmux)
## Note-Used Only for testing or developement 
### login to your instance
```
ssh -i your_key.pem ubuntu@<your_public_ip>
```
### install tmux
```
sudo apt update && sudo apt install tmux -y
```
### start tmux session #tmux new -s disease_treatment_app
```
tmux new -s disease_treatment_app
```
### Run Your API
```
cd ~/disease_treatment
source disease_treatment_venv/bin/activate   # if you have a virtual environment
gunicorn --bind 0.0.0.0:5000 app:app --log-level info --capture-output --enable-stdio-inheritance
```
### Detach the session (keep it running in background)
Ctrl + B, then D

### Reattach anytime to view logs
tmux attach -t disease_treatment_app

### Stop the app
Reattach (tmux attach -t disease_treatment_app)
Press Ctrl + C to stop
Type exit to close tmux
