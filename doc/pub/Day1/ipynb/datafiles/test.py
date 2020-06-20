import pandas as pd

data = "EoS.csv"

Points = pd.read_fwf(data, usecols= (1,2),
                    names=('One', 'Two'),
                    index_col=False)

 
data=pd.read_csv("EoS.csv")

data.sort_values(['One'], axis=0, ascending=True, inplace=True)
