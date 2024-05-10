import json

import requests
import streamlit as st


from db_client import Searcher

CLIP_URL = "http://127.0.0.1:8000"
E5_URL = "http://127.0.0.1:8001"

@st.cache_resource()
def get_db(index_name):
    return Searcher(index_name=index_name)

def handle_image(query, input_type, k=10):

    if input_type == "text":
        response = requests.post(CLIP_URL + "/texts", json={ "texts": [query] })
    elif input_type == "image":
        pass
        # image = utils.load_image_from_url(query)
        # query_features = clip_model.get_image_features(image)
    else:
        return []
    db = get_db("image")
    data = response.json()
    query_features = data["feature_vectors"]
    response = db.knn_search(query_features[0], k=k)
    return response["hits"]["hits"]

def handle_chats(query, k=10):
    response = requests.post(E5_URL + "/query", json={ "texts": [query] })
    data = response.json()
    query_features = data["feature_vectors"]
    db = get_db("chats")
    response = db.knn_search(query_features[0], k=k)
    return response["hits"]["hits"]

def display_image_results(results):
    n_cols = 3
    cols = st.columns(n_cols)

    # Reset / empty
    for col in cols:
        col.empty()
    
    # Populate
    for idx, hit in enumerate(results):
        image_url = hit["_source"]["url"] + "?w=360"
        score = hit["_score"]
        cols[idx % n_cols].image(image_url, f"Score:{score:.2f}")

def display_chats_results(results):

    table_data = [{
        "text" : doc["_source"]["text"] ,
        "score" : doc["_score"]
    } for doc in results]
    st.table(table_data)


def main():
    st.set_page_config(page_title="Image Search Engine")

    st.title("Vector Search Engine")

    # Sidebar
    k_results = st.sidebar.slider("# Results:", value=10, min_value=1, max_value=100)
    
    # Image Search
    st.sidebar.header("Unsplash Image Search")
    input_type = st.sidebar.radio("Query by", ("text", "image"))
    query_image = st.sidebar.text_input("Enter text/image URL here:")
    submit_image = st.sidebar.button("Submit image/text query")
    
    # Chat Search
    st.sidebar.header("Toxic Chat Search")
    query_chat = st.sidebar.text_input("Enter text here:")
    submit_chat = st.sidebar.button("Submit chat query")

    if submit_image:

        if not query_image:
            st.sidebar.error("Please enter a query.")
            return
        
        results = handle_image(query_image, input_type, k=k_results)
        display_image_results(results)

    elif submit_chat:
        if not query_chat:
            st.sidebar.error("Please enter a query.")
            return
        
        results = handle_chats(query_chat, k=k_results)
        display_chats_results(results)
    else:
        st.write("""
    The database contains 25,000 images from the Unsplash Dataset. You can either

    - search them using a natural language description (e.g., animals in jungle), or
    - find similar images by providing an image URL (e.g. https://i.imgur.com/KRNOn22.jpeg).

    The algorithm will return the ten most relevant images.
    """)

if __name__ == "__main__":
    main()