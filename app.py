import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.layers import LSTM, Dense


# Importando dados do diretorio Base/CSV

caminho = "./Base/CSV/"
arquivos = os.listdir(caminho)
dados = []
for csv in arquivos:
    filepath = caminho + csv
    d = pd.read_csv(filepath, sep="\t")
    dados.append(d)

dados = pd.concat(dados)

# Divisao em conjunto de teino e teste
train, test = train_test_split(dados, test_size=0.3, random_state=569)
train.insert(0, 'CONJUNTO', 'TRN')
test.insert(0, 'CONJUNTO', 'TST')

dados = pd.concat([test, train])

# Extraçao de dados em CSV
# dados.to_csv('pre_processamento.csv', sep='\t')

# Pre-Processamento
CinemasA = [ 'Espaço Itaú de Cinema Pompéia',
'Espaço Itaú de Cinema Frei Caneca',
'Espaço Itaú de Cinema Brasília',
'Cineart Cidade',
'Cineart Del Rey',
'Cineart Boulevard Shopping']

CinemasB = ['Petra Belas Artes',
'Cinemas Teresina',
'Espaço Itaú de Cinema Rio de Janeiro']

CinemasC = [
    "Cineart Shopping Contagem",
    "Moviecom Maxi Shopping",
    "Moviecom Praia Shopping",
    "Cinemais Uberaba",
    "Cineart Monte Carmo",
    "Cineplus Jardim das Américas",
    "Moviecom Shopping Castanheira",
    "Moviecom Pátio",
    "Cineart Minas Shopping",
    "Moviecom Unimart Cinemas",
    "Moviecom Shopping Buriti",
    "Cineart ItaúPower",
    "Roxy 5",
    "Cinemais Jardim Norte – Juiz de Fora",
    "Circuito Bonsucesso Shopping",
    "Moviecom Shopping Penha",
    "Grupo Cine Itapetininga",
    "Moviemax Rosa e Silva",
    "Grupo Cine Eusébio",
    "Moviecom Shopping do Vale",
    "Moviecom Macapá Shopping",
    "Topázio Cinemas Polo Shopping Indaiatuba",
    "Moviecom Pátio Marabá",
    "Moviecom Taubaté Shopping Center",
    "Cinemais Anápolis",
    "CineX Shopping Total",
    "Cineart Via Shopping"
]

CinemasD = [
    "Espaço de Cinema Bourbon Country",
    "Espaço Itaú de Cinema Augusta",
    "Cinemas Multiplex Topázio",
    "Cine A Norte Sul",
    "Centerplex Boulevard",
    "Grupo Cine Rio Claro",
    "Centerplex Shopping Center Limeira",
    "Planet Cinemas Niteroi",
    "Cine A Marabá",
    "Una Cine Belas Artes",
    "CineX Palmas Shopping",
    "Cine Metha Glauber Rocha"
]

CinemasE = [
    "Cinemais Guaratinguetá",
    "Moviecom Páteo Itaquá",
    "Centerplex Pátio Norte Shopping",
    "Grupo Cine Sete Lagoas",
    "Centerplex Aracaju",
    "Cineart Serra Sul Shopping",
    "Roxy 6",
    "Moviecom Shopping Boa vista",
    "Cinemais Montes Claros",
    "Grupo Cine Pindamonhangaba",
    "Centerplex Via Sul Shopping",
    "Moviecom Shopping Tivoli",
    "Moviemax Camará Shopping",
    "Centerplex Grand Shopping Messejana",
    "Grupo Cine Itapecerica",
    "Circuito Cinemas Penápolis",
    "Circuito Ponta Porã",
    "Centerplex Pátio Maceió",
    "Centerplex Suzano",
    "CM Cinemas Rio das Ostras",
    "Grupo Cine Vitória de Santo Antão",
    "Grupo Cine Várzea Paulista",
    "Centerplex Serramar Shopping",
    "Cineplus Xaxim",
    "Grupo Cine Andradina",
    "Grupo Cine Pacajus",
    "Cineart Barbacena",
    "Circuito Parauapebas",
    "Centerplex Grande Circular",
    "Planet Shopping Difusora",
    "Grupo Cine Valinhos",
    "Multicine Santa Maria Shopping",
    "Grupo Cine Guarujá",
    "Grupo Cine Cajamar",
    "Centerplex Caruaru Shopping",
    "Moviecom Conquista Sul",
    "Cine Guedes",
    "Centerplex North Shopping Maracanaú",
    "Grupo Cine Embu das Artes",
    "Grupo Cine Tucuruí",
    "Cine Vip.com",
    "Grupo Cine Amargosa",
    "Cinemais Patos de Minas",
    "Grupo Cine Maranguape",
    "Cine Esmeralda",
    "Moviemax Igarassu Shopping",
    "Centerplex Carpina",
    "Centerplex Cidade Norte",
    "Centerplex Poços de Caldas",
    "Cine Itaim Paulista",
    "Grupo Cine Diadema",
    "MaxiMovie Saquarema",
    "Multicine Águas Lindas",
    "Grupo Cine Fernandópolis",
    "Grupo Cine Videira",
    "CineX Luziânia Shopping",
    "Cinemais Bougainville",
    "Multicine Caxias Shopping",
    "Cinemaxxi Cidade Luz",
    "Moviecom Shopping Jaú",
    "Multicine Iandê Shopping",
    "Grupo Cine Santa Inês",
    "Cinemais Lorena",
    "Centerplex Itapevi Center",
    "Moviemax Eldorado",
    "Cineplus Campo Largo",
    "Centerplex Cascavel",
    "Circuito Cinemas Karajás Shopping",
    "Cineplus Fazenda Rio Grande",
    "Cineplus Castro",
    "Cinemais Ituiutaba",
    "Cinemais Araxá",
    "Multicine Cidade Jardim",
    "Multicine Brasil Center Shopping",
    "Cine Bom Vizinho Itapipoca",
    "Cine Bom Vizinho Quixadá",
    "Cine Francisco Lucena",
    "Roxy 3",
    "Cine Bom Vizinho Acaraú"
]

CinemasF = [
    "Cine A Resende",
    "Moviecom PrudenShopping",
    "Centerplex Mag Shopping",
    "Cine A Café Alfenas",
    "Cine A Bragança",
    "Moviecom Franca Shopping",
    "Multicine Mossoró",
    "Cine A Café Araras",
    "Cineart Ponteio Lar Shopping",
    "Cine A Rio Verde",
    "Cine A Altamira",
    "Multicine Sobral",
    "Cine A Café Pouso Alegre",
    "IBICinemas",
    "Moviecom Shopping Jaraguá",
    "CineX Garden Shopping",
    "Cinemaxx Mercado Estação",
    "Cine A Itajubá",
    "Centerplex North Shopping Barretos",
    "Cine 3 Ferry Boat´s Plaza",
    "CM Cinemas Bacaxá",
    "CM Cinemas Araruama",
    "Cine Lume Ritz",
    "Cine A Café São Lourenço",
    "Cine A Café Lavras",
    "Cine A São João da Boa Vista",
    "Cine A Café Três Corações",
    "Cine Teatro Vila Rica",
    "Cine A Café Itapira",
    "Multicine Parnaíba",
    "Cinen Fun",
    "Grupo Cine Rio do Sul",
    "Cine Bom Vizinho Sobral",
    "MaxiMovie Unamar",
    "Cine A Serra Talhada",
    "Cine Cavaliere Orlandi",
    "Multicine Patos Shopping",
    "Espaço Itaú Botafogo",
    "Multicine Jataí",
    "Cine A Mineiros",
    "Cine Atibaia",
    "Grupo Cine Santa Isabel",
    "Movie L´América Shopping",
    "Movie Arte Shopping Bento",
    "Multiplex Itatiba Mall",
    "Centro Cultural Unimed-BH Minas",
    "Cine A Continental",
    "Cine São José Jequié",
    "Cine Mococa"
]

cinemas = {'A': CinemasA, 
           'B': CinemasB, 
           'C': CinemasC, 
           'D': CinemasD, 
           'E': CinemasE, 
           'F': CinemasF 
           }

cinema_escala = []

for cinema_nome in dados['Cinema']:
    indice = next((chave for chave, valor in cinemas.items() if cinema_nome in valor), 'G')
    cinema_escala.append(indice)

dados['CinemaEscala'] = cinema_escala

ExibidorA = ['AFA Cinemas', 'Belas Artes Grupo', 'Cine 14 Bis', 'Cine Bom Vizinho']
ExibidorB = ['Centerplex Cinemas', 'Cine 3 Ferry Boat´s Plaza', 'Cine A', 'Cine Belas Artes']
ExibidorC = ['Cine Cavaliere Orlandi', 'Cine Center', 'Cine Clacita', 'Cine Cocais', 'Cine GACEMSS', 'Cine Globo Cinemas']
ExibidorD = [
    'Cine Esmeralda',
    'Cine Guedes',
    'Cine Itaim Paulista',
    'Cine Kimak',
    'Cine Lume',
    'Cine Lume Ritz',
    'Cine Lúmine',
    'Cine Metha Glauber Rocha'
]

exibidores = {
    'A': ExibidorA,
    'B': ExibidorB,
    'C': ExibidorC,
    'D': ExibidorD
}

exibidor_escala = []

for exibidor_nome in dados['Exibidor']:
    indice = next((chave for chave, valor in exibidores.items() if exibidor_nome in valor), 'E')
    exibidor_escala.append(indice)

dados['ExibidorEscala'] = exibidor_escala

dados['SINOPSE'].fillna('N/A', inplace=True)

dados['generoList'] = [generos.split(',') for generos in dados['GENERO']]

genero_A = ['Drama', 'Animação']
genero_B = ['Ação', 'Fantasia,Aventura,Ação', 'Ação,Aventura', 'Aventura']
genero_C = ['Terror', 'Comédia,Aventura,Ação', 'Comédia', 'Suspense']
genero_D = ['Animação, Aventura, Comédia', 'Ação, Aventura, Fantasia, Ficção-científica', 'Ação, Aventura, Fantasia', 'Comédia, Aventura, Animação']
genero_E = ['Ação, Fantasia, Terror', 'Drama, Aventura, Ação', 'Ação, Drama, Fantasia', 'Ação, Comédia, Terror']
genero_F = ['Terror, Suspense', 'Terror, Thriller', 'Família', 'Ficção-científica, Comédia, Aventura, Ação', 'Fantasia, Família, Aventura', 'Aventura, Família, Fantasia']
genero_G = ['Policial, Ação',
    'Romance',
    'Animação, Família',
    'Ficção-científica, Aventura, Ação',
    'Desenho, Infantil',
    'Documentário',
    'Esporte, Drama',
    'Família, Infantil',
    'Família, Comédia, Animação',
    'Ação, Aventura, Comédia',
    'Ficção',
    'Aventura, Animação',
    'Drama, Romance',
    'Biografia',
    'Aventura, Drama, Ficção-científica',
    'Romance, Drama',
    'Drama, Comédia'
]
genero_H = [
    'Ação, Aventura, Crime',
    'Terror, Suspense, Ficção-científica',
    'Ação, Comédia, Drama',
    'Ação, Aventura, Suspense',
    'Fantasia, Ação',
    'Aventura, Ação',
    'Ação, Animação, Aventura',
    'Aventura, Animação, Ação'
]

generos = {
    'A': genero_A,
    'B': genero_B,
    'C': genero_C,
    'D': genero_D,
    'E': genero_E,
    'F': genero_F,
    'G': genero_G,
    'H': genero_H,
}

genero_escala = []

for genero in dados['GENERO']:
    indice = next((chave for chave, valor in generos.items() if genero in valor), 'I')
    genero_escala.append(indice)

dados['GeneroEscala'] = genero_escala

dumie = pd.get_dummies(dados['GeneroEscala'], prefix='Genero')
dados = pd.concat([dados, dumie], axis=1)

distribuidor_A = ['Universal Pictures', 'Walt Disney Studios',
             'Sony Pictures', 'Warner Bros.', 'Paramount Pictures']
distribuidor_B = ['Paris Filmes', 'Diamond Films',
             'Downtown/Paris', 'Imagem Filmes']
distribuidor_C = [
    'PlayArte Pictures',
    'Pandora Filmes',
    'O2 Play',
    'California Filmes',
    'Imovision',
    'Diamond/Galeria',
    'Imagem/Califórnia',
    'Embaúba Filmes',
    'Arteplex Filmes',
    'Elo Company/H2O Films'
]

distribuidor_D = [
    'Bonfilm',
    'Vitrine Filmes',
    'Cinecolor Films Brasil',
    'Galeria Distribuidora',
    'Synapse Distribution'
]

distribuidor = {
    'A': distribuidor_A,
    'B': distribuidor_B,
    'C': distribuidor_C,
    'D': distribuidor_D,
}

distribuidor_escala = []

for dist in dados['Distribuidora']:
    indice = next((chave for chave, valor in distribuidor.items() if dist in valor), 'E')
    distribuidor_escala.append(indice)

dados['DistribuidorEscala'] = distribuidor_escala

def removeStr(x):
    if isinstance(x, str):
        remove = ['<i>Vozes de:</i>', '<br>', '<i>Vozes de: </i>']
        for y in remove:
            x = x.replace(y, '')
        return x
    else:
        return x
    
dados['ELENCO'] = dados['ELENCO'].apply(removeStr)

dumie = pd.get_dummies(dados['CinemaEscala'], prefix='Cinema')
dados = pd.concat([dados, dumie], axis=1)

dumie = pd.get_dummies(dados['ExibidorEscala'], prefix='Exibidor')
dados = pd.concat([dados, dumie], axis=1)

dumie = pd.get_dummies(dados['DistribuidorEscala'], prefix='Distribuidor')
dados = pd.concat([dados, dumie], axis=1)
dados['Legendado'] = [x if x == 'L' else 0 for x in dados['SessaoLegenda'] ]
dados['Nacional'] = [x if x == 'N' else 0 for x in dados['SessaoLegenda'] ]

dados['SessaoVideo'] = dados['SessaoVideo'].str.strip()
dados['3D'] = [1 if '3D' in x else 0 for x in dados['SessaoVideo']]

dados['salas_A'] = [1 if x in [4, 3] else 0 for x in dados['QTD_SALAS']]
dados['salas_B'] = [1 if x in [5, 6] else 0 for x in dados['QTD_SALAS']]
dados['salas_C'] = [1 if x in [1, 2] else 0 for x in dados['QTD_SALAS']]
dados['salas_D'] = [1 if x in [7, 8, 9, 11] else 0 for x in dados['QTD_SALAS']]

dados['ESTREIA'] = pd.to_datetime(dados['ESTREIA'])
dados['CinesemanaInicio'] = pd.to_datetime(dados['CinesemanaInicio'])

dados['semana'] = dados['CinesemanaInicio'].dt.isocalendar().week
dados['ano'] = pd.DatetimeIndex(dados['CinesemanaInicio']).year
dados['MesEstreia'] = pd.DatetimeIndex(dados['ESTREIA']).month
dados['AnoEstreia'] = pd.DatetimeIndex(dados['ESTREIA']).year

dados['QtdGenero'] = [x.count(',') + 1 for x in dados['GENERO']]

dados['DiasEmExibicao'] = (dados['ESTREIA'] - dados['CinesemanaInicio']).abs().dt.days



data = dados[['CONJUNTO',
              'Brasileiro',
              'ClassificacaoIndicativa',
              'Duracao',
              'Comercial',
              'Cinema_A',
              'Cinema_B',
              'Cinema_C',
              'Cinema_D',
              'Cinema_E',
              'Cinema_F',
              'Exibidor_A',
              'Exibidor_B',
              'Exibidor_C',
              'Exibidor_D',
              'salas_A',
              'salas_B',
              'salas_C',
              'salas_D',
              'Genero_A',
              'Genero_B',
              'Genero_C',
              'Genero_D',
              'Genero_E',
              'Genero_F',
              'Genero_G',
              'Genero_H',
              'Distribuidor_A',
              'Distribuidor_B',
              'Distribuidor_C',
              'Distribuidor_D',
              'QtdGenero',
              'semana',
              'ano',
              'MesEstreia',
              'AnoEstreia',
              'DiasEmExibicao',
              'QtdSessoes'
              ]]

data = data.fillna(-1)

X_treino = data.query("CONJUNTO == 'TRN'")
X_teste = data.query("CONJUNTO == 'TST'")

y_treino = X_treino['QtdSessoes']
y_teste  = X_teste['QtdSessoes']

del X_treino['QtdSessoes']
del X_teste['QtdSessoes']

del X_treino['CONJUNTO']
del X_teste['CONJUNTO']

#########################################################
####                  MLP Regressor                  ####
#########################################################
print('\n\nMLP Regressor\n\n')
modelo = MLPRegressor(
    activation='relu',
    hidden_layer_sizes=(10, 20),
    random_state=9,
    alpha=0.001,
    early_stopping=False
    )

indice = list(range(0,len(y_teste)))

modelo.fit(X_treino, y_treino)



pred = modelo.predict(X_teste)

test_set_rsquared = modelo.score(X_teste, y_teste)
test_set_rmse = np.sqrt(mean_squared_error(y_teste, pred))
#
# Print R_squared and RMSE value
#
print('R_squared value: ', test_set_rsquared, '\nRMSE:', test_set_rmse, '\nErro Médio Absoluto:', mean_absolute_error(y_teste, pred))



#########################################################
####                  Random Forest                  ####
#########################################################
print('\n\nRandom Forest\n\n')
modelo = RandomForestRegressor(n_estimators=20, 
                               min_samples_leaf=4, 
                               min_impurity_decrease=0.0004,
                               n_jobs=-1,
                               random_state=42) 
modelo.fit(X_treino, y_treino)


predicoes = modelo.predict(X_teste)

mse = mean_squared_error(y_teste, predicoes)  # Erro Quadrático Médio
mae = mean_absolute_error(y_teste, predicoes)  # Erro Médio Absoluto

print('Erro Quadrático Médio (MSE):', mse)
print('Erro Médio Absoluto (MAE):', mae)

# Obtém a importância das características (features) do modelo
importancias_features = modelo.feature_importances_

# Obtém os nomes das características (features)
nomes_features = X_treino.columns

# Ordena as importâncias das features em ordem decrescente
indices = importancias_features.argsort()[::-1]

# Plota as importâncias das características
plt.figure(figsize=(10, 6))
plt.title("Importância das Características (Features)")
plt.bar(range(X_treino.shape[1]), importancias_features[indices], align="center")
plt.xticks(range(X_treino.shape[1]), [nomes_features[i] for i in indices], rotation=90)
plt.xlim([-1, X_treino.shape[1]])
plt.tight_layout()
plt.show()


#########################################################
####               Exportar Diagrama                #####
#########################################################
###   Execute somente se estiver com tempo de sobra   ###
#########################################################

#from sklearn.tree import export_graphviz
#import graphviz

## Escolha um índice de árvore específico da floresta (por exemplo, 0 para a primeira árvore)
#indice_arvore = 0

## Exporte a árvore para um arquivo .dot
#export_graphviz(modelo.estimators_[indice_arvore], out_file='tree.dot', 
#                feature_names = X_treino.columns,
#                rounded = True, proportion = False, 
#                precision = 2, filled = True)#

# Converta o arquivo .dot em um gráfico
#with open("tree.dot") as f:
#    dot_graph = f.read()
#graphviz.Source(dot_graph)

#########################################################
####                     XGboost                     ####
#########################################################
print('\n\nXGBoost\n\n')
dtrain = xgb.DMatrix(X_treino, label=y_treino)
dtest = xgb.DMatrix(X_teste, label=y_teste)

params = {
    "max_depth": 3,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "objective": "reg:squarederror",  # Para problemas de regressão
}

model = xgb.XGBRegressor(**params)
model.fit(X_treino, y_treino)

predictions = model.predict(X_teste)

rmse = mean_squared_error(y_teste, predictions, squared=False)
r2 = r2_score(y_teste, predictions)
ema = mean_absolute_error(y_teste, predictions)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")
print(f"Erro médio Absoluto: {ema}")

#########################################################
####     Exportar Rasultados Para Analise Manual     ####
#########################################################

final = dados.query("CONJUNTO == 'TST'")
final['predicoes'] = predicoes
final['DifAbs'] = (final['QtdSessoes']-final['predicoes'])*-1
final.to_csv('./resultados.csv', sep=';')

#########################################################
####                      LSTM                       ####
#########################################################

print('\n\nLSTM\n\n')
# Normalização dos dados
scaler_x = MinMaxScaler()
X_treino_norm = scaler_x.fit_transform(X_treino)
X_teste_norm = scaler_x.transform(X_teste)

scaler_y = MinMaxScaler()
y_treino_norm = scaler_y.fit_transform(np.array(y_treino).reshape(-1, 1))
y_teste_norm = scaler_y.transform(np.array(y_teste).reshape(-1, 1))

# Redimensionamento dos dados para LSTM
X_treino_lstm = X_treino_norm.reshape((X_treino_norm.shape[0], 1, X_treino_norm.shape[1]))
X_teste_lstm = X_teste_norm.reshape((X_teste_norm.shape[0], 1, X_teste_norm.shape[1]))

# Construção do modelo LSTM
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_treino_lstm.shape[1], X_treino_lstm.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Camada de saída

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_treino_lstm, y_treino_norm, epochs=100, batch_size=32, verbose=1)

# Avaliação do modelo
train_loss = model.evaluate(X_treino_lstm, y_treino_norm, verbose=0)
test_loss = model.evaluate(X_teste_lstm, y_teste_norm, verbose=0)

print(f'Treino - Loss: {train_loss:.4f}')
print(f'Teste - Loss: {test_loss:.4f}')

# Obter as previsões do modelo para os dados de treino e teste
y_pred_treino = model.predict(X_treino_lstm)
y_pred_teste = model.predict(X_teste_lstm)

# Calcular o MAE para os dados de treino e teste
train_mae = mean_absolute_error(y_treino, y_pred_treino)
test_mae = mean_absolute_error(y_teste, y_pred_teste)

print(f'Treino - MAE: {train_mae:.4f}')
print(f'Teste - MAE: {test_mae:.4f}')


print('\n\nFIM\n\n')























