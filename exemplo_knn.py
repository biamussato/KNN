import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import numpy as np

data = [
    {"id": 1, "titulo": "Matrix", "generos": ["Ação", "Ficção Científica"], "popularidade": 8.7},
    {"id": 2, "titulo": "Vingadores", "generos": ["Ação", "Aventura"], "popularidade": 8.0},
    {"id": 3, "titulo": "O Poderoso Chefão", "generos": ["Drama", "Crime"], "popularidade": 9.2},
    {"id": 4, "titulo": "Interestelar", "generos": ["Ficção Científica", "Drama"], "popularidade": 8.6},
    {"id": 5, "titulo": "Gladiador", "generos": ["Ação", "Drama"], "popularidade": 8.5},
    {"id": 6, "titulo": "Titanic", "generos": ["Romance", "Drama"], "popularidade": 7.9},
    {"id": 7, "titulo": "O Senhor dos Anéis", "generos": ["Fantasia", "Aventura"], "popularidade": 9.0},
    {"id": 8, "titulo": "Harry Potter", "generos": ["Fantasia", "Aventura"], "popularidade": 8.1},
    {"id": 9, "titulo": "Coringa", "generos": ["Drama", "Crime"], "popularidade": 8.4},
    {"id": 10, "titulo": "Deadpool", "generos": ["Ação", "Comédia"], "popularidade": 8.0},
    {"id": 11, "titulo": "Logan", "generos": ["Ação", "Drama"], "popularidade": 8.2},
    {"id": 12, "titulo": "Pantera Negra", "generos": ["Ação", "Ficção Científica"], "popularidade": 7.8},
    {"id": 13, "titulo": "Django Livre", "generos": ["Faroeste", "Drama"], "popularidade": 8.4},
    {"id": 14, "titulo": "O Lobo de Wall Street", "generos": ["Comédia", "Crime"], "popularidade": 8.2},
    {"id": 15, "titulo": "Clube da Luta", "generos": ["Drama", "Suspense"], "popularidade": 8.8},
    {"id": 16, "titulo": "Star Wars: Uma Nova Esperança", "generos": ["Ficção Científica", "Aventura"], "popularidade": 8.6},
    {"id": 17, "titulo": "Velozes e Furiosos", "generos": ["Ação", "Crime"], "popularidade": 7.2},
    {"id": 18, "titulo": "It: A Coisa", "generos": ["Terror", "Suspense"], "popularidade": 7.5},
    {"id": 19, "titulo": "Invocação do Mal", "generos": ["Terror", "Suspense"], "popularidade": 7.9},
    {"id": 20, "titulo": "Toy Story", "generos": ["Animação", "Aventura"], "popularidade": 8.3},
    {"id": 21, "titulo": "Shrek", "generos": ["Animação", "Comédia"], "popularidade": 8.0},
    {"id": 22, "titulo": "Divertida Mente", "generos": ["Animação", "Família"], "popularidade": 8.1},
    {"id": 23, "titulo": "Frozen", "generos": ["Animação", "Fantasia"], "popularidade": 7.4},
    {"id": 24, "titulo": "Procurando Nemo", "generos": ["Animação", "Aventura"], "popularidade": 8.2},
    {"id": 25, "titulo": "Mad Max: Estrada da Fúria", "generos": ["Ação", "Ficção Científica"], "popularidade": 8.1}
]

df = pd.DataFrame(data)

mlb = MultiLabelBinarizer()
generos_encoded = mlb.fit_transform(df["generos"])
generos_df = pd.DataFrame(generos_encoded, columns=mlb.classes_)

scaler = MinMaxScaler()
df["popularidade_norm"] = scaler.fit_transform(df[["popularidade"]])

X = pd.concat([generos_df, df[["popularidade_norm"]]], axis=1)

knn = NearestNeighbors(n_neighbors=3, metric="euclidean")
knn.fit(X)

def recomendar_filmes_knn(preferencias_usuario, popularidade_usuario, num_recomendacoes=3):

    preferencias_usuario = [g.strip().capitalize() for g in preferencias_usuario]
    genero_usuario = [1 if g in preferencias_usuario else 0 for g in mlb.classes_]
    popularidade_usuario_norm = scaler.transform([[popularidade_usuario]])[0][0]
    entrada_usuario = np.array(genero_usuario + [popularidade_usuario_norm]).reshape(1, -1)
    distancias, indices = knn.kneighbors(entrada_usuario, n_neighbors=num_recomendacoes)

    recomendacoes = df.iloc[indices[0]][["titulo", "generos", "popularidade"]]
    return recomendacoes

entrada_generos = input("Opçõoes de gêneros:\n"
                        "Ação\n"
                        "Animação\n"
                        "Aventura\n"
                        "Comédia\n"
                        "Crime\n"
                        "Drama\n"
                        "Fantasia\n"
                        "Família\n"
                        "Faroeste\n"
                        "Ficção Científica\n"
                        "Romance\n"
                        "Suspense\n"
                        "Terror\n"
                        "Digite os gêneros desejadoos separados por vírgula (ex: ação, drama): ")
entrada_popularidade = float(input("Digite a popularidade desejada: "))

preferencias_usuario = entrada_generos.split(",")

print("\n Filmes recomendados:")
print(recomendar_filmes_knn(preferencias_usuario, entrada_popularidade))
