import requests
import os
from dotenv import load_dotenv

def obtener_noticias_baloncesto(prompt):
    # Carga las variables de entorno desde el archivo .env
    load_dotenv()

    NEWS_API_KEY = os.environ["NEWS_API_KEY"]

    url = "https://newsapi.org/v2/everything"

    parametros = {
        "q": "baloncesto",
        "sortBy": "publishedAt",
        "apiKey": NEWS_API_KEY,
    }
    respuesta = requests.get(url, params=parametros)
    noticias = respuesta.json()

    resultado = ""

    for noticia in noticias["articles"][:5]:
        resultado += f'- Título: {noticia["title"]} \n - Descripción: {noticia["description"]} \n - URL: {noticia["url"]} \n --- \n\n'

    return resultado


# obtener_noticias_baloncesto()
