
import pandas as pd

df = pd.read_csv('D:/Capstone/cresci-2017/processed_data.csv')
print(df['label'].value_counts())
