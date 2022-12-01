import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from sklearn.metrics.pairwise import linear_kernel
import json
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


from sklearn.feature_extraction.text import TfidfVectorizer

netflix_data = pd.read_csv('netflix_titles.csv')

#Filling null values with empty string.
filledna=netflix_data.fillna('')

#Cleaning the data - making all the words lower case
def clean_data(x):
        return str.lower(x.replace(" ", ""))

#Identifying features on which the model is to be filtered.
features=['title','director','cast','listed_in','description']
filledna=filledna[features]

for feature in features:
    filledna[feature] = filledna[feature].apply(clean_data)

def create_soup(x):
    return x['title']+ ' ' + x['director'] + ' ' + x['cast'] + ' ' +x['listed_in']+' '+ x['description']

filledna['soup'] = filledna.apply(create_soup, axis=1)
tfidf  = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filledna['soup'])


cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
filledna=filledna.reset_index()
indices = pd.Series(netflix_data.index, index=netflix_data['title']).drop_duplicates()
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return netflix_data.iloc[movie_indices]

#print(get_recommendations('Dark'))
#tfidf_matrix.shape
def Table(df):
    fig=go.Figure(go.Table( columnorder = [1,2,3],
          columnwidth = [10,28],
            header=dict(values=[' Title','Description'],
                        line_color='black',font=dict(color='black',size= 22),height=50,
                        fill_color='#424281',
                        align=['left','center']),
                cells=dict(values=[df.title,df.description],
                       fill_color='#EFF2F5',line_color='grey',
                           font=dict(color='black', family="Lato", size=16),height=50,
                       align='left')))
    fig.update_layout(height=350, title ={'text': "Top 10 Movie Recommendations", 'font': {'size':25}},title_x=0.7
                     )
    return st.plotly_chart(fig,use_container_width=True)
movie_list = netflix_data['title'].values


####################################################################
#streamlit
##################################################################

st.set_page_config(
    page_title="Netflix Movie Recommendation System",
    page_icon="film_frames",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.header("🎞️ Netflix Movie Recommendation System")
lottie_coding = load_lottiefile("m4.json")
st_lottie(
    lottie_coding,
    speed=1,
    reverse=False,
    loop=True,
    quality="low",height=350
)
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names = get_recommendations(selected_movie)
    #list_of_recommended_movie = recommended_movie_names.to_list()
   # st.write(recommended_movie_names[['title', 'description']])
    Table(recommended_movie_names)
    
st.write('  '
         )
st.write(' ')

git = st.checkbox('Show Netflix Data Source')
if git :
    st.write(
        "check out this [link](https://github.com/hummetbelli/Netflix_Recommendation_System)")
