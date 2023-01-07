import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import seaborn as sns


def main():
    data = pd.read_csv("/Users/zhuguanyu/Desktop/scrubbed.csv")
    data_new = data.drop(['date posted', 'comments', 'duration (hours/min)', 'country', 'state'], axis=1)
    print(data_new.isnull().sum())
    print(data_new.describe(include='all').round(2))
    print(data_new.info())
    # Now we see 300 is the most frequent value and we can use it to fill in the null value (data imputation)
    print(data_new['duration (seconds)'].value_counts())
    # Only focus on the dataframe where shape is not null, cause shape is very important considering UFO
    new_data = data_new[data_new['shape'].notnull()]
    # Change some data types
    new_data['shape'] = new_data['shape'].astype('category')
    new_data['city'] = new_data['city'].astype('category')
    new_data['datetime'] = data['datetime'].apply(my_datetime)
    # Error = coerce means returning NaN (means Null in pandas) if encountering wrong format value
    new_data['duration (seconds)'] = pd.to_numeric(data['duration (seconds)'], errors='coerce')
    new_data['latitude'] = pd.to_numeric(data['latitude'], errors='coerce')
    new_data['longitude '] = pd.to_numeric(data['longitude '], errors='coerce')
    # new_data.loc[:, 'latitude'].replace({0.0: np.nan}, inplace=True)
    new_data.loc[new_data['latitude'].isnull(), 'latitude'] = 33.200088
    # new_data.loc[:, 'longitude '].replace({0.0: np.nan}, inplace=True)
    new_data.loc[new_data['duration (seconds)'].isnull(), 'duration (seconds)'] = 300
    # new_data.loc[:, 'duration (seconds)'].replace({0.0: np.nan}, inplace=True)
    new_data['month'] = pd.DatetimeIndex(new_data['datetime']).month
    new_data['year'] = pd.DatetimeIndex(new_data['datetime']).year
    new_data['day'] = pd.DatetimeIndex(new_data['datetime']).day
    new_data['hour'] = pd.DatetimeIndex(new_data['datetime']).hour
    new_data = new_data.drop(['datetime'], axis=1)

    df_final = new_data.copy()

    # Standardization (clusters need standardization, choose only numeric values)
    for i in df_final.select_dtypes(exclude='category').columns:
        df_final.loc[:, i] = StandardScaler().fit_transform(np.array(df_final[[i]]))

    print(df_final.info())

    # Find which cluster number that costs the less (though number up, cost goes down, so deciding by my intuition)

    # K = range(1, 6)
    # cost = []
    # for k in K:
    #      kproto = KPrototypes(n_clusters=k, init='Cao', random_state=42)
    #      kproto.fit_predict(df_final, categorical=[0, 1])
    #      cost.append(kproto.cost_)
    # print(cost)

    # Apply K-Prototypes Clustering and set 5 clusters with categorical value: city, shape
    kproto = KPrototypes(n_clusters=5, init='Cao', n_jobs=4)
    clusters = kproto.fit_predict(df_final, categorical=[0, 1])
    # Combine original dataset with the labeling cluster number
    df_clusters = pd.concat([df_final, pd.DataFrame({'cluster': clusters})], axis=1)

    print(df_clusters.head())

    print(df_clusters.info())

    # df_clusters = df_clusters.astype({"duration (seconds)": 'int64', "latitude": 'int64', "longitude ": 'int64',
    #                             "month": 'int64', "year": 'int64', "day": 'int64', "hour": 'int64'},
    #                            errors='ignore')

    # sns.set(rc={'axes.facecolor': 'black', 'figure.facecolor': 'black', 'axes.grid': False})

    # for i in df_clusters:
    #     g = sns.FacetGrid(df_clusters, col="cluster", hue="cluster", palette="Set2")
    #     g.map(plt.hist, i, bins=10, ec="k")
    #     g.set_xticklabels(rotation=30, color='white')
    #     g.set_yticklabels(color='white')
    #     g.set_xlabels(size=15, color='white')
    #     g.set_titles(size=15, color='#FFC300', fontweight="bold")
    #     g.fig.set_figheight(5)


def my_datetime(date):
    if date[11:13] == '24':
        x = date[:11] + '23:59'
    elif date[10:12] == '24':
        x = date[:10] + '23:59'
    elif date[9:11] == '24':
        x = date[:9] + '23:59'
    else:
        return pd.to_datetime(date)
    return pd.to_datetime(x)


if __name__ == '__main__':
    main()
    
# Reference: https://www.kaggle.com/code/miguelfzzz/store-customers-clustering-analysis