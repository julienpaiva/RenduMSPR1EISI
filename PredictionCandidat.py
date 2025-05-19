import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.compose import ColumnTransformer
from tensorflow.keras.optimizers import Adam

# Charger et préparer les données, on drop la colonne 'nom' puisque c'est la cible et la colonne 'sexe' puisqu'elle n'apporte rien
data = pd.read_csv('donneespropre.csv', sep=';')
X = data.drop(columns=['nom', 'sexe'])
y = data['nom']

# Identifier les colonnes catégorielles
categorical_cols = X.select_dtypes(include=['object']).columns

# Créer un transformateur de colonnes pour encoder les variables catégorielles
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'  # Laisser les autres colonnes inchangées
)

# Appliquer le transformateur aux données
X = preprocessor.fit_transform(X)

# On Divise les données en ensembles d'entraînement et de test en ratio 80 / 20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardiser les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir les étiquettes en one-hot encoding
y_train = to_categorical(y_train, num_classes=11)
y_test = to_categorical(y_test, num_classes=11)

# Construire le modèle de réseau neuronal avec une architecture simplifiée
model = Sequential()
# 1ere couche, couche d'entrée
model.add(Dense(6, input_dim=X_train.shape[1], activation='relu'))
# 2eme couche, couche intermediaire
model.add(Dense(4, activation='relu'))
# 3eme couche, couche de sortie
model.add(Dense(11, activation='softmax'))

# Ici on compile le modèle
#optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ici on entraine le modèle sur 30 cycle
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32, verbose=1)

# Ici on évalue le modèle et récupere la précision
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Précision du modèle : {test_accuracy:.2f}')

# On charge de nouvelles données/échantillons
new_data = pd.read_csv('newdata.csv', sep=';')
X_new = new_data.drop(columns=['nom', 'sexe'])

# On transforme les données en valeurs numériques pour les colonnes comportant des valeurs 'string' puis on les standardise
X_new = preprocessor.transform(X_new)
X_new = scaler.transform(X_new)

#On lance la prédiction ici avec les nouvelles données
predictions = model.predict(X_new)

# Convertir les prédictions en étiquettes de classe
predicted_classes = predictions.argmax(axis=-1)

# On affiche les prédictions du modèle
print(predicted_classes)