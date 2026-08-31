# Data

The datasets used to train the models in this project aren't included in this repo (too large for GitHub). Download them from the sources below and place them here before running any training scripts.

## Email dataset

**Source:** [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) (Kaggle)

Save as:

data/phishing_email_dataset.csv


Expected columns: 'sender', 'receiver', 'date', 'subject', 'body', 'label', 'urls'

## URL dataset

**Source:** [PhiUSIIL Phishing URL Dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) (UCI Machine Learning Repository)

Save as:

data/PhiUSIIL_Phishing_URL_Dataset.csv


## After downloading

Run the cleaning/training scripts in 'ML/' to regenerate the processed datasets and model files:


ML/audit_url_dataset.py       # cleans the URL dataset
ML/train_email_model.py       # cleans + trains the email model


These write their output into 'data/processed/' and 'models/', which also aren't tracked in this repo.