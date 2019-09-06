import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from sklearn import preprocessing
from keras.layers import Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

def get_data(path):
    df = pd.read_csv(path)
    return df


def cat_num(data):
    df = data
    categorical_cols=df.columns[df.dtypes==object].tolist()
    le = LabelEncoder()
    df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
    return df

def preprocess_data(data):
    df = data
    df[df.drop('income', axis=1).columns] = preprocessing.scale(df.drop('income', axis=1))
    return df

def scaling_data(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(data)
    normalized = scaler.transform(data)
    normalized_data=pd.DataFrame(normalized,columns=list(data.columns))
    return normalized_data

def train_test_split_data(df):
    x = df.drop('income', axis=1).to_numpy()
    y = df['income']
    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def base_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = accuracy_score(y_test, pred)
    return score


def learning_curve_plot(model, X_train, y_train, title_name):
    train_sizes, train_scores, valid_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    train_scores_mean = - train_scores.mean(axis=1)
    validation_scores_mean = -valid_scores.mean(axis=1)
    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('error', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title(title_name, fontsize=18, y=1.03)
    plt.legend()
    plt.savefig("./img/"+title_name+".png")
    plt.close()


def freq_dist(df):
    df.drop('income', axis=1).hist(bins=50, figsize=(20, 15))
    plt.savefig("./img/frequency_dist.png", orientation='potrait')
    plt.close()


def base_neural_model(X_train, X_test, y_train, y_test):
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model = Sequential([
        Dense(400, activation='relu',
              input_dim=X_train.shape[1], kernel_initializer='random_normal'),
        Dense(400, activation='relu', kernel_initializer='random_normal'),
        Dense(300, activation='relu', kernel_initializer='random_normal'),
        Dense(2, activation='softmax', kernel_initializer='random_normal'), ])
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(
        X_train,
        to_categorical(y_train),
        epochs=100,
        batch_size=10,
    )
    model.evaluate(
        X_test,
        to_categorical(y_test)
    )
    model.save('./model/my_model.h5')
