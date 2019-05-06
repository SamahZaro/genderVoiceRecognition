from model import *
import pandas as pd
import numpy as np

checkpoint_path = "../saved_model"
mymodel, att_cols, mean, std = load_model(checkpoint_path)

df= pd.read_csv("R/voice.csv")
df = preprocess_data(df, normalize = True, att_cols = att_cols, mean=mean, std=std)
test_sample = df.sample(frac=1).reset_index(drop=True)
test_input = test_sample.loc[:, att_cols]
test_label = test_sample.loc[:, 'label_num']
  
print("sample: ", test_sample.to_string())

prediction = predict(mymodel, test_input)

print("predicted label num\treal label num")
for idx, sample in test_sample.iterrows():
	print(prediction[idx],"\t\t\t", test_label[idx])

class_mapping = {0:'female', 1:'male'}
print("where: ", class_mapping)
