import pandas as pd
import os

# File is in a folder relative to your current working directory
# data = pd.read_csv(os.getcwd() + "/deepspeech_transcript/csv_docs/paths.csv")
# print(data)
# data = pd.read_csv("../csv_docs/paths.csv")
# print(data)
cwd = os.getcwd()+"/deepspeech_transcript/csv_docs/paths.csv"
print("Current working directory: ",cwd)
print("PATH: ")
print(os.getcwd())
