import pandas as pd
import os

# File is in a folder relative to your current working directory
data = pd.read_csv(os.getcwd() + "/deepspeech_transcript/csv_docs/paths.csv")
print(data)

# cwd = os.getcwd()+"\\deepspeech_transcript"
# print("Current working directory: ",cwd)
