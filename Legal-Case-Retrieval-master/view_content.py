import pandas as pd
import pprint

data = pd.read_csv("./dataset/dataset.csv")
print("Finished reading dataset.")
while(True):
	number = int(input())
	content = data.loc[data['document_id'] == number].content
	pprint.pprint(list(content.values)[0])
