import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


def main():
    # Book title + book score
    books = pd.read_csv('/Users/zhuguanyu/Desktop/goodbooks_10k_rating_and_description.csv', index_col=0)
    books = books.reset_index()
    # Book ratings
    ratings = pd.read_csv('/Users/zhuguanyu/Desktop/ratings.csv', index_col=0)
    ratings = ratings.reset_index()
    # Recent reading/ renting books (user_id & book_id)
    read = pd.read_csv('/Users/zhuguanyu/Desktop/to_read.csv', index_col=0)
    read = read.reset_index()
    # Select user who already read more than 40 books
    select_user = ratings["user_id"].value_counts()
    # Get series's index
    select_user = select_user[select_user > 40].index
    # Filter based on rating and score to minimize dataset
    rows = (books["book_rating"] > 4.3) & (books["book_score"] > books["book_score"].median())
    books = books[rows]
    # Get series's values
    select_book_id = books['book_id'].values
    rows2 = (read['user_id'].isin(select_user)) & (read['book_id'].isin(select_book_id))
    read = read[rows2]
    # Combine dataset to get user_id + title
    new_read = read.merge(books[["book_id", "title"]], on="book_id")
    new_read[["Quantity"]] = 1
    # Unstack data to get grouped user as row and book titles as columns
    new_read_apr = new_read.groupby(["user_id", "title"])["Quantity"].sum().unstack().fillna(0)
    # To prepare our data so that it corresponds to the format (1/0) of association rule
    new_read_apr = new_read_apr.applymap(change)
    # Apply Association Rule
    frequent_items = apriori(new_read_apr, min_support=0.002, use_colnames=True)
    result = association_rules(frequent_items, metric="lift", min_threshold=1)
    result = result.sort_values(by="confidence", ascending=False)
    # Only take ante, cons and conf
    result = pd.concat([result["antecedents"], result["consequents"], result["confidence"]], axis=1).reset_index()
    # Get Ante and Cons's lengths as there may be more than one
    result["antecedents_length"] = result["antecedents"].apply(lambda x: len(x))
    result["consequents_length"] = result["consequents"].apply(lambda x: len(x))
    # Ramdonly select the book you want as input (ex: The Book Thief), and then get the recommendations!
    result = result[result["antecedents"].str.contains("The Book Thief", regex=False)].sort_values("confidence", ascending=False).head(10)
    print(result)


def change(x):
    if x <= 0:
        return 0
    elif x >= 1:
        return 1


if __name__ == '__main__':
    # See all columns in Console
    pd.set_option('display.max_columns', None)
    main()

# References:
# https://www.kaggle.com/code/baturalpsert/association-rfm-analysis
# https://www.kaggle.com/code/mustafayazici/association-rule-learning-recommendation-system/notebook