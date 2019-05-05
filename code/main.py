from model import *
import pandas as pd

checkpoint_path = "../saved_model"
mymodel = load_model(checkpoint_path)

df= pd.read_csv("../data/voice.csv")

df, att_cols,_ ,_ = preprocess_data(df)
test_sample = df.sample( n = 1 ).reset_index(drop=True)

test_input = test_sample.loc[:, att_cols]
test_label = test_sample.loc[:, 'label_num']
  
print("sample: ", test_sample.to_string(index=False))

prediction = predict(mymodel, test_input)
print("predicted label num.: ", prediction)
print("real label num: ", test_label.to_string(index=False))
predicted_label = "male" if (prediction == 1) else "female"
print("predicted label: ", predicted_label)