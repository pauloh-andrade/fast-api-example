import os
import pickle

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


iris = datasets.load_iris()
dados = pd.DataFrame(data=iris.data, columns = iris.feature_names)

def mapear_nomes(numero_classe):
    return(iris.target_names[numero_classe])


dados['flower'] = iris.target
dados['flower_name'] = dados['flower'].apply(mapear_nomes)

x = dados.drop(columns = ['flower', 'flower_name']) # Somente Comprimento do Abdômen e Comprimento das Antenas
y = dados['flower_name']                # Classe alvo
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 8)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_predicoes = logreg.predict(x_test)

logreg.predict(np.array([5.0, 3.6, 1.4, 0.2]).reshape(1, -1))

filename = 'finalized_model.pkl'
pickle.dump(logreg, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
loaded_model.predict(np.array([5.0, 3.6, 1.4, 0.2]).reshape(1, -1))

print(result)