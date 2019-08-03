import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
df = pd.read_csv('USA_Housing.csv')
plot = sns.scatterplot(x='Avg. Area Income',y='Price',data=df)
plt.show(sns)