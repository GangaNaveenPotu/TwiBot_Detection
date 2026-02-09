# TwiBot Detection Project

This project is for detecting bots on Twitter.

## Setup

1. Clone the repository.
2. The following files and directories have been ignored from this repository to reduce the size. Please download them and place them in the correct directories.

### Ignored Files and Directories:

- **cresci-2017 dataset:**
  - `cresci_2017_project/cresci-2017/edge.csv`
  - `cresci_2017_project/cresci-2017/label.csv`
  - `cresci_2017_project/cresci-2017/node.json`
  - `cresci_2017_project/cresci-2017/split.csv`
- **Twibot-20 dataset:**
  - `twibot_20_project/Twibot-20 kaggle dataset/dev.json`
  - `twibot_20_project/Twibot-20 kaggle dataset/test.json`
  - `twibot_20_project/Twibot-20 kaggle dataset/train.json`
- **Pre-trained Model:**
  - The `bert-base-uncased` directory contains the pre-trained model. You can download it from Hugging Face.
- **Processed Data:**
  - `twibot20_processed_data.csv`
  - `cresci-2017/processed_data.csv`
- **Results:**
  - The `results/`, `cresci_2017_project/results/`, and `twibot_20_project/results/` directories contain the results of the model training and testing. These will be generated when you run the models.

After placing the datasets in the correct folders, you can run the `data_preparation.py` and `twibot20_data_preparation.py` scripts to process the data. Then you can run the models.
