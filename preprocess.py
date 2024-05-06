import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('stopwords')

def main():
    try:
        df = pd.read_csv('diabetes.csv')
        if df.empty:
            print("The CSV file is empty.")
            return

        nonext = os.path.splitext('diabetes.csv')[0]

        df = preprocess(df)

        # corr_matrix = df.corr()

        # plt.figure(figsize=(10,8))
        # sns.heatmap(corr_matrix, annot=True, fmt='.2f', cbar=True)
        # plt.title('Pearson Correlation of Non-Textual Features')
        # plt.show()

        x = df.drop(['diabetes', 'smoking_history', 'gender'], axis=1)
        y = df['diabetes']

        print(x)

        x.to_csv('diabetes_x.csv', index=False)
        y.to_csv('diabetes_y.csv', index=False)


    # if something is wrong with csv file
    except Exception as e:
        print(f"An error occured: {e}")


def preprocess(df):
    try:
        # df = pd.get_dummies(df, columns=['gender', 'smoking_history'], drop_first=True)
        #one hot encode categorical
        # df = pd.get_dummies(df, columns=['gender'], drop_first=True)

        #ensure these are binary
        df['hypertension'] = df['hypertension'].astype(int)
        df['heart_disease'] = df['heart_disease'].astype(int)

        #scale numeric features
        scaler = StandardScaler()
        numeric_feats = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
        df[numeric_feats] = scaler.fit_transform(df[numeric_feats])

        return df

    except Exception as e:
        print(f"An error occured: {e}")


def stopTok(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    return filtered_tokens


if __name__ == "__main__":
    main()
