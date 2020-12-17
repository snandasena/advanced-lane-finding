## How to run this project
### Step01: Clone or download this project 
Note: Before run this project you can see final results from output directory or by opening provided HTML file in a browser.

### Step02: Running this project using Python3.6 or above version
#### Create a Python virtual enviroment
Note: All commands were tested only Ubuntu 16.04
```bash
# Install virtualenv
sudo apt install virtualenv
# Create a Python virtual machine
virtualenv -p python3.6 ~/pvm
# Activate PVM
source ~/pvm/bin/activate
# Install dependencies
python -m pip install -r requirments.txt
# Run Jupyer notebook
jupyter notebook <location to IPython notebook file>
```
