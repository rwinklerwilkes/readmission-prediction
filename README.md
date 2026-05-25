Source: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

To run Jupyter within uv:
`uv run --with jupyter jupyter lab`

To run FastAPI within uv:
`uv run fastapi dev`

There are two other directories that are needed to get this running:
* data: This contains the following datasets:
> * diabetic_data.csv, downloaded from https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
> * icd9.txt, an IC9 mapping downloaded from https://github.com/drobbins/ICD9/blob/master/icd9.txt
> * IDS_mapping.csv, downloaded from https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
* models: This gets created via the model.py notebook in production/. These are too large to save to Github so I have models/ in my .gitignore