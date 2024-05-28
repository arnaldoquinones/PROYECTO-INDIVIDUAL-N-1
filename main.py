from fastapi.responses import JSONResponse
from fastapi import FastAPI
import pandas as pd
import numpy as np  
#from sklearn.utils.extmath           import randomized_svd
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# columnstouse=['item_id','playtime_forever','user_id']
df_games = pd.read_parquet(r'D:\Users\Arnaldo\Desktop\SISTEMAS\SOYHENRY\CURSO\PROYECTOS\PROYECTO INDIVIDUAL I\RECURSADO\ARCHIVOS\DATAFRAMES\DATAFRAMES PARKET\df_games.parquet')
df_items = pd.read_parquet(r'D:\Users\Arnaldo\Desktop\SISTEMAS\SOYHENRY\CURSO\PROYECTOS\PROYECTO INDIVIDUAL I\RECURSADO\ARCHIVOS\DATAFRAMES\DATAFRAMES PARKET\df_items.parquet') # columns=columnstouse
df_reviews = pd.read_parquet(r'D:\Users\Arnaldo\Desktop\SISTEMAS\SOYHENRY\CURSO\PROYECTOS\PROYECTO INDIVIDUAL I\RECURSADO\ARCHIVOS\DATAFRAMES\DATAFRAMES PARKET\df_reviews.parquet')
df_reviews_coments = pd.read_parquet(r'D:\Users\Arnaldo\Desktop\SISTEMAS\SOYHENRY\CURSO\PROYECTOS\PROYECTO INDIVIDUAL I\RECURSADO\ARCHIVOS\DATAFRAMES\DATAFRAMES PARKET\df_reviews_coments.parquet')

# df_games = df_games.head(14000)
# df_items = df_items.head(14000)
# df_reviews = df_reviews.head(14000)
# df_reviews_coments = df_reviews_coments.head(14000)

app=FastAPI()

# http://127.0.0.1:8000

@app.get("/")
def index():
    return "Hola, bienvenido(a) a mi proyecto"

@app.get('/desarrolladora/{desarrollador}')
def developer(desarrollador: str):
    desarrollador_lower = desarrollador.lower()
    df = df_games.loc[df_games['developer'].str.lower() == desarrollador_lower].sort_values('year')
    df = df.drop_duplicates()
    
    yearB = 0
    dict_años = defaultdict(list)
    items = 0
    items_free = 0
    
    for index, row in df.iterrows():
        year = row['year']
        price = row['price']
        
        if year != yearB:
            items = 0  # Restablece la cuenta de elementos si es un nuevo año
            yearB = year
            items_free = 0
        
        items += 1
        if price == 0.00:
            items_free += 1
        
        dict_años[year] = [items, items_free]
    
    texto = ' Contenido Free (%)'
    column_name_free = f'{desarrollador}{texto}'
    
    # Crear el DataFrame
    df_result = pd.DataFrame(dict_años.values(), index=dict_años.keys(), columns=['Cantidad de Items', column_name_free])
    
    # Calcular el porcentaje de contenido free
    df_result[column_name_free] = round((df_result[column_name_free] / df_result['Cantidad de Items'] * 100), 2)
    
    return df_result

####
# Código necesario para el funcionamiento de la función userdata()
# Explicación en el notebook 06_Funciones.ipynb
# Calcular la media de items_count por user_id
df_items_count = df_items.groupby("user_id")['items_count'].mean().reset_index()

# Seleccionar las columnas user_id y recommend
result = df_reviews[['user_id', 'recommend']]

# Asegurarse de que la columna recommend sea de tipo float
result['recommend'] = result['recommend'].astype(float)

# Calcular la media de recommend por user_id y redondear a 2 decimales
df_user_recommend_ratio = result.groupby('user_id')['recommend'].mean().round(2).reset_index()

print(df_user_recommend_ratio)
####

@app.get('/datos_por_usuario/{user_id}')
def userdata(User_id: str):
    # Filtrar los DataFrames por el User_id
    user_reviews = df_reviews[df_reviews['user_id'] == User_id]
    user_items = df_items[df_items['user_id'] == User_id]
    
    # Calcular cantidad de dinero gastado
    # Este es un placeholder, debes ajustar la lógica de cálculo según tus datos
    # Por ejemplo, si hay una columna que indique el costo de cada item
    dinero_gastado = user_items['playtime_forever'].sum() * 0.1  # Esto es solo un ejemplo
    
    # Calcular el porcentaje de recomendación
    if len(user_reviews) > 0:
        porcentaje_recomendacion = (user_reviews['recommend'].sum() / len(user_reviews)) * 100
    else:
        porcentaje_recomendacion = 0
    
    # Calcular la cantidad de items
    cantidad_items = user_items['items_count'].sum()
    
    # Crear el diccionario con los resultados
    resultado = {
        "Usuario X": User_id,
        "Dinero gastado": f"{dinero_gastado:.2f} USD",
        "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
        "cantidad de items": cantidad_items
    }
    
    return resultado

@app.get('/usuario_por_genero/{genero}')
def UserForGenre(genero: str):
    global df_games
    global df_items

    # Se crea un nuevo DataFrame llamado df_games que contiene solo las columnas relevantes
    game_columns = ['item_id', 'app_name', 'year', 'Indie', 'Action', 'Casual', 'Adventure', 'Strategy',
                    'Simulation', 'RPG', 'Free to Play', 'Early Access', 'Sports',
                    'Massively Multiplayer', 'Racing', 'Design &amp; Illustration', 'Utilities']
    df_games_filtered = df_games[game_columns]
    
    # Se hace un join entre df_games y df_items utilizando 'item_id' como clave
    df_joined = pd.merge(df_games_filtered, df_items[['item_id', 'playtime_forever', 'user_id']], on='item_id', how='inner')

    # Se filtra el DataFrame resultante para el género específico
    genre_df = df_joined[df_joined[genero] == 1]

    # Se agrupa por usuario y año, sumamos las horas jugadas y se encuentra el usuario con más horas jugadas para el género dado
    total_hours_by_user_and_year = genre_df.groupby(['user_id', 'year'])['playtime_forever'].sum()
    max_user = total_hours_by_user_and_year.groupby('user_id').sum().idxmax()

    # Se obtiene la lista de acumulación de horas jugadas por año para el usuario con más horas jugadas
    max_user_hours_by_year = total_hours_by_user_and_year.loc[max_user].reset_index()
    max_user_hours_list = [{"Año": int(row['year']), "Horas": row['playtime_forever']} for _, row in max_user_hours_by_year.iterrows()]

    # Se retorna el resultado en un formato específico
    result = {"Usuario con más horas jugadas para Género {}".format(genero): max_user, "Horas jugadas": max_user_hours_list}
    return result

@app.get('/top3_desarrolladoras_por_año/{anho}')
def best_developer_year(año: int):
    # Filtrar los juegos del año proporcionado en df_games
    juegos_del_año = df_games[df_games['year'] == año]

    # Obtener los IDs de los juegos del año
    ids_juegos_año = juegos_del_año['item_id']

    # Filtrar las reseñas de los juegos del año con recomendaciones positivas y análisis de sentimiento positivo en df_reviews
    reseñas_recomendadas = df_reviews[(df_reviews['item_id'].isin(ids_juegos_año)) & 
                                      (df_reviews['recommend'] == True) & 
                                      (df_reviews['sentiment_analysis'] == 2)]

    # Hacer un join entre df_reviews y df_games utilizando 'item_id' como clave
    merged_reviews = pd.merge(reseñas_recomendadas, df_games[['item_id', 'developer']], on='item_id', how='left')

    # Calcular la frecuencia de cada desarrollador
    developer_count = merged_reviews['developer'].value_counts()

    # Seleccionar el top 3 de desarrolladores más recomendados, ordenados de mayor a menor por value_counts
    top_3_best_developers = developer_count.head(3)

    # Retornar el resultado en un formato específico
    result = [{"Puesto {}".format(i+1): developer} for i, (developer, _) in enumerate(top_3_best_developers.items())]
    
    return result

@app.get('/Analisis_reseña_desarrollador/{desarrolladora}')
def developer_reviews_analysis(developer: str):
    # Filtrar df_games por el desarrollador especificado
    filtered_games = df_games[df_games['developer'] == developer]

    # Unir los datos filtrados con df_reviews utilizando la columna 'item_id'
    merged_data = pd.merge(filtered_games, df_reviews, on='item_id')

    # Filtrar los datos para excluir los sentimientos neutrales
    merged_data = merged_data[merged_data['sentiment_analysis'].isin([0, 2])]

    # Mapear los valores de sentiment_analysis a 'Positive' y 'Negative'
    sentiment_mapping = {0: 'Negative', 2: 'Positive'}
    merged_data['sentiment_analysis'] = merged_data['sentiment_analysis'].map(sentiment_mapping)

    # Contar las reseñas por tipo de sentimiento
    sentiment_counts = merged_data['sentiment_analysis'].value_counts()

    # Obtener los recuentos de positivos y negativos
    positive_count = sentiment_counts.get('Positive', 0)
    negative_count = sentiment_counts.get('Negative', 0)

    # Crear el diccionario con el resultado
    result = {developer: {'Negative': negative_count, 'Positive': positive_count}}

    return result


####
# Código necesario para el funcionamiento de la función recomendacion_juego()
# Explicación en el notebook 05_Modelo_item.ipynb
# Se crea un nuevo DataFrame llamado df_games_final que contiene solo las columnas relevantes
game_columns = ['item_id', 'app_name']
df_games_final = df_games[game_columns]

# Se hace un join entre df_games y df_UserItems utilizando 'item_id' como clave
df_joined = pd.merge(df_games_final, df_reviews_coments[['item_id', 'review']], on='item_id', how='inner')

#Se crea una muestra para el modelo
muestra = df_joined.head(20000)

# Se crea el modelo de machine learning con Scikit-Learn
tfidf = TfidfVectorizer(stop_words='english')
muestra=muestra.fillna("")

tdfid_matrix = tfidf.fit_transform(muestra['review'])
cosine_similarity = linear_kernel( tdfid_matrix, tdfid_matrix)
####

@app.get('/Recomendación_juegos/{id}')
def recomendacion_juego(id: int):
    if id not in muestra['item_id'].values:
        return {'mensaje': 'No existe el id del juego'}
    
    titulo = muestra.loc[muestra['item_id'] == id, 'app_name'].iloc[0]
    idx = muestra[muestra['app_name'] == titulo].index[0]
    sim_cosine = list(enumerate(cosine_similarity[idx]))
    sim_scores = sorted(sim_cosine, key=lambda x: x[1], reverse=True)
    sim_ind = [i for i, _ in sim_scores[1:6]]
    sim_juegos = muestra['app_name'].iloc[sim_ind].values.tolist()
    return {f'juegos recomendados para {id}': list(sim_juegos)}


# Test
# print("-----------------------------")
# print(developer("Id Software"))
# print("-----------------------------")
# print(userdata("--ionex--"))
# print("-----------------------------")
# print(UserForGenre("Casual"))
# print("-----------------------------")
# print(best_developer_year(2015))
# print("-----------------------------")
# print(developer_reviews_analysis("Valve"))
# print("-----------------------------")
# print(recomendacion_juego(2300))
