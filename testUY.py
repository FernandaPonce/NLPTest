# CORPUS: un conjunto de textos, ordenados o no, que sirven de base para cualquier análisis lingüístico o estadístico.
# Utilizamos el corpus creado por la Sociedad Española de Procesamiento del Lenguaje Natural, en su TASS: Taller de Análisis Semántico en la SEPLIN,
# su objetivo original era el avance de la investigación sobre análisis de sentimientos en español (http://tass.sepln.org/)

import spacy
from spacy import displacy
#Stop Words de es_core_news_sm
from spacy.lang.es.stop_words import STOP_WORDS
nlp = spacy.load('es_core_news_sm')

# eliminar las stopwords : conectores (no anaden más contenido a la frase, preposiciones, conjunciones, etc)
stopwords_spacy = list(STOP_WORDS)

#Clasificacion de texto
#Importar las librerías necesarias para leer los archivos de TASS
import xmltodict
import json
import pandas as pd
import re

#Traer el archivo xml original xml y convertirlo en un diccionario
with open("TASS2019_country_UY_train.xml", encoding="UTF-8") as xml_file:
    data_dict = xmltodict.parse(xml_file.read())
xml_file.close()

#Convertir a json el diccionario
json_data = json.dumps(data_dict)

#Escribir en un archivo el resultado en json
with open("TASS2019_country_UY_train.json", "w") as json_file:
    json_file.write(json_data)
json_file.close()

#Limpieza de datos en la fuente
#Original en json
fin = open("TASS2019_country_UY_train.json", "rt")
#Archivo resultante en json
fout = open("TASS2019_country_UY_train-sintilde.json", "wt")
#Procesamiento de lìneas del archivo

for line in fin:
	#Reemplazar los caracteres unicode, no se dejaron tildes porque causan error
    strtmp1 = line.replace('\\u00f1', 'ñ')
    strtmp1 = strtmp1.replace('\\u00e1', 'a')
    strtmp1 = strtmp1.replace('\\u00e9', 'e')
    strtmp1 = strtmp1.replace('\\u00ed', 'i')
    strtmp1 = strtmp1.replace('\\u00f3', 'o')
    strtmp1 = strtmp1.replace('\\u00fa', 'u')
    strtmp1 = strtmp1.replace('\\u00bf', '¿')
    strtmp1 = strtmp1.replace('\\u00a1', '¡')
    strtmp1 = strtmp1.replace('\\u00d1', 'Ñ')
    strtmp1 = strtmp1.replace('\\u00c1', 'A')
    strtmp1 = strtmp1.replace('\\u00c9', 'E')
    strtmp1 = strtmp1.replace('\\u00cd', 'I')
    strtmp1 = strtmp1.replace('\\u00d3', 'O')
    strtmp1 = strtmp1.replace('\\u00da', 'U')
    strtmp1 = strtmp1.replace('\\u00fc', 'ü')
    strtmp1 = strtmp1.replace('\\u00b0', '')
    #Quitar el inicio y el fin del json para dejar solo los tweets
    strtmp1 = strtmp1.replace('{"tweets": {"tweet": ', '')
    strtmp1 = strtmp1.replace(']}}', ']')
    #Quitar el diccionario que contiene la polaridad y dejarla solo con su valor de sentimiento
    strtmp1 = strtmp1.replace('"sentiment": {"polarity": {"value": ', '"sentiment": ')
    strtmp1 = strtmp1.replace('"NONE"}}', '"NONE"')
    #Asignamos al sentimiento positivo el valor de 1
    strtmp1 = strtmp1.replace('"P"}}', '1')
    strtmp1 = strtmp1.replace('"NEU"}}', '"NEU"')
    #Asignamos al sentimiento negativo el valor de 0
    strtmp1 = strtmp1.replace('"N"}}', '0')
    #eliminación de puntuaciones
    strtmp1 = re.sub('[¡!#$).;¿?&°]', '', strtmp1.lower()) #tmb poner datos en minuscula
    fout.write(strtmp1)
#cerrar archivos
fin.close()
fout.close()

#tomar los datos del archivo creado en un dataframe
train_df = pd.read_json('TASS2019_country_UY_train-sintilde.json', encoding="ISO-8859-1")
train_df.head()
#print(train_df)

#Función para eliminar las menciones a otros usuarios de twitter
def filter_reply(content):
    temp = content
    while temp.find("@") > -1:
        temp = temp[:temp.find("@")] + temp[(temp.find(" ",temp.find("@"))):]
    return temp

#Quitar menciones del texto
train_df['content'] = train_df['content'].apply(filter_reply)

#Quitar columnas sin clasificación de sentimiento
indexNames = train_df[(train_df['sentiment'] == 'none') | (train_df['sentiment'] == 'neu')].index
train_df.drop(indexNames, inplace=True)
train_df.head()
print(train_df)

#Importar librerías de aprendizaje
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Verificar frecuencias de cada categoría
print(train_df['sentiment'].value_counts())

#Verificar si hay datos nulos
#print(train_df.isnull().sum())

#Tokenizacion

#Constante de signos de puntuación (para referencia ya que se eliminaron en el archivo fuente)
import string
puntua = string.punctuation + '¡¿'
#print(puntua)

# Función para limpieza de datos
def text_data_cleaning(sentence):
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.strip()
        else:
            temp = token
        tokens.append(temp)

    clean_tokens = []
    for token in tokens:
        if token not in stopwords_spacy and token not in puntua:
            clean_tokens.append(token)

    return clean_tokens

#Vectorization Feature Engineering (TF-IDF)

#importar librería de vectorización
from sklearn.svm import LinearSVC

#Definir la función de tokenizado y crear el clasificador
tfidf = TfidfVectorizer(tokenizer=text_data_cleaning)
classifier = LinearSVC()

#Crear los vectores de datos
X = train_df['content']
y = train_df['sentiment']

#Crear el vector de entrenamiento como una porción de los datos y dejar el resto para pruebas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#print(X_train.head())

#Crear un pipeline
clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])

#evitar que el formato se tome como unknown
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#Entrenar el clasificador
clf.fit(X_train, y_train)

#Crear el vector de valores predichos a partir del clasificador
y_pred = clf.predict(X_test)

#Ver la precisión obtenida
print(classification_report(y_test, y_pred))

#Crear la matriz de confusión
confusion_matrix(y_test, y_pred)

#Predecir algunas frases de prueba

example1="Me gustó mucho el recital"
example2="Esto es violencia de genero"

#print(clf.predict(['Realmente me gustó mucho este ejercicio']))
if (clf.predict([example1])) == [1]:
    print("La frase: " + example1 + " es POSITIVA")

#Una negativa
#print(clf.predict(['violencia']))
if (clf.predict([example2])) == [0]:
    print("La frase: " + example2 + " es NEGATIVA")





