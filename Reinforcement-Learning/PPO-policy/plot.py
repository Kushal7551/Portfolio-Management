import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv('btc_6h.csv')
# p=(len(data)//4)*3
# data=data[49878:85533]
plt.plot(data['Close']*100000/data['Close'].iloc[0])
plt.savefig("fig10.png")