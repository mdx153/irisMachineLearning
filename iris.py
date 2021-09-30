from sklearn.tree import DecisionTreeClassifier
import pandas as pd

base = pd.read_csv('iris.csv')

x = base.iloc[:,1:5].values
y = base.iloc[:,5].values

x.shape
y.shape

modelo = DecisionTreeClassifier()
modelo.fit(x,y)

    
#exemplo
previsao = modelo.predict([[0.4,0.5,8.0,3.0]])
print(previsao)