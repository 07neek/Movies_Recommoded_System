#from matplotlib.pyplot import stem
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# class IMDbRecommender:
#     def __init__(self, csv_path):
#         self.data = pd.read_csv("C:\\Users\\Neek\\Downloads\\imdb_top_1000.csv", usecols=["Series_Title", "Released_Year", "IMDB_Rating", "No_of_Votes"])
#         self._preprocess()
#         self._build_similarity()

#     def _preprocess(self):
#         self.data = self.data.dropna(subset=["Series_Title", "Released_Year", "IMDB_Rating", "No_of_Votes"])
#         self.data["Released_Year"] = pd.to_numeric(self.data["Released_Year"], errors="coerce")
#         self.data = self.data.dropna()

#         scaler = MinMaxScaler()
#         self.data[["Released_Year", "IMDB_Rating", "No_of_Votes"]] = scaler.fit_transform(
#             self.data[["Released_Year", "IMDB_Rating", "No_of_Votes"]]
#         )

#     def _build_similarity(self):
#         tfidf = TfidfVectorizer(stop_words="english")
#         tfidf_matrix = tfidf.fit_transform(self.data["Series_Title"])

#         numeric_matrix = self.data[["Released_Year", "IMDB_Rating", "No_of_Votes"]].values
#         combined_matrix = np.hstack([tfidf_matrix.toarray(), numeric_matrix])

#         self.similarity_matrix = cosine_similarity(combined_matrix)

#     def recommend(self, movie_title, top_k=10):
#         if movie_title not in self.data["Series_Title"].values:
#             raise ValueError("Movie not found in dataset")

#         idx = self.data[self.data["Series_Title"] == movie_title].index[0]
#         sim_scores = list(enumerate(self.similarity_matrix[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)


#         top_indices = [i for i, _ in sim_scores[1:top_k+1]]
#         return self.data.iloc[top_indices][["Series_Title", "Released_Year", "IMDB_Rating", "No_of_Votes"]]

#     def precision_at_k(self, k=10, sample_size=50):
#         np.random.seed(42)
#         sampled_movies = np.random.choice(self.data["Series_Title"].values, size=sample_size, replace=False)

#         precision_scores = []
#         for movie in sampled_movies:
#             try:
#                 recs = self.recommend(movie, top_k=k)
#                 # Ground truth: movies with similar release year ±2 years considered "relevant"
#                 target_year = self.data.loc[self.data["Series_Title"] == movie, "Released_Year"].values[0]
#                 relevant = self.data[np.abs(self.data["Released_Year"] - target_year) <= 0.05]  # 0.05 scaled ≈ 2 yrs
#                 relevant_titles = set(relevant["Series_Title"].values)

#                 retrieved = set(recs["Series_Title"].values)
#                 precision = len(retrieved & relevant_titles) / k
#                 precision_scores.append(precision)
#             except:
#                 continue
            
#         if not precision_scores:
#             return "TRY AGAIN"

#         return np.mean(precision_scores) if precision_scores else 0.0


# if __name__ == "__main__":
#     recommender = IMDbRecommender("imdb_top_1000.csv")

#     print("Sample Recommendations for 'The Shawshank Redemption':")
#     print(recommender.recommend("The Shawshank Redemption", top_k=5))

#     print("\nEvaluating Precision@10 ...")
#     score = recommender.precision_at_k(k=10, sample_size=50)
#     print(f"Precision@10 (approx): {score:.3f}")



class MovieRecommonded:
    def __init__(self):
        self.df = pd.read_csv("C:\\Users\\Neek\\Downloads\\imdb_top_1000.csv", usecols=["Series_Title", "Released_Year", "Genre", "Overview", "Star1", "Star2", "Star3"])

    def show_data(self):
        self.df['Star'] = self.df['Star1'] + ',' + self.df['Star2'] + ',' + self.df['Star3']
        self.df['Star'] = self.df['Star'].str.lower()
        self.new_df = self.df[["Series_Title", "Released_Year", "Genre", "Overview", "Star"]].copy()

        remove_words = r'\b(the|and|his|her|on|or|a|to|from|in)\b'
        self.new_df['Overview'] = (
            self.new_df['Overview']
            .str.replace(remove_words, '', case=False, regex=True)
            .str.replace(' +', ' ', regex=True)
            .str.strip()
            .str.lower()
        )
        self.new_df['Genre'] = self.new_df['Genre'].str.lower()

        self.new_df['content'] = (
            self.new_df['Overview'] + " " +
            self.new_df['Genre'] + " " +
            self.new_df['Star']
        )    
        #print(self.new_df)
        #merge three columns in one columns)

    def vectorization(self):
        self.stem()
        cnt = CountVectorizer(max_features=2000, stop_words='english')
        self.vec = cnt.fit_transform(self.new_df['content']).toarray()
        # x = cnt.get_feature_names_out()
        # print(self.vec)
        # print(x)

    @staticmethod
    def stem_text(text):
        ps = PorterStemmer()
        return ' '.join([ps.stem(i) for i in text.split()])

    def stem(self):
        self.new_df['content'] = self.new_df['content'].apply(MovieRecommonded.stem_text)

    def similarity(self):
        self.similarity = cosine_similarity(self.vec)


    def recommended(self, movie):
        matches = self.new_df[self.new_df['Series_Title'] == movie]
        if matches.empty:
            print("Movie not found.")
            return
        movie_index = matches.index[0]
        dis = self.similarity[movie_index]
        mov_list = sorted(list(enumerate(dis)), reverse=True, key=lambda x: x[1])[1:6]
        print(f"Top 5 recommendations for {movie}:")
        for i in mov_list:
            print(self.new_df.iloc[i[0]]['Series_Title'])
            print(i[1])

if __name__ == '__main__':
    recommender = MovieRecommonded()
    recommender.show_data()
    recommender.vectorization()
    recommender.similarity()
    print(recommender.recommended("Interstellar"))

    # This will print the first 5 rows



