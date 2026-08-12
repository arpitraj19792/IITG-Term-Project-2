import streamlit as st
import pandas as pd
import json
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Basic page setup
st.set_page_config(page_title="🎬 Anime Recommender", layout="wide")

# Mapping genres so they look consistent
GENRE_MAPPING = {
    "shonen": "Shonen", "shoujo": "Shojo", "shojo": "Shojo", "seinen": "Seinen", "josei": "Josei",
    "kodomo": "Kodomo", "action": "Action", "adventure": "Adventure", "comedy": "Comedy", "drama": "Drama",
    "fantasy": "Fantasy", "sci-fi": "Sci-Fi", "science fiction": "Sci-Fi", "romance": "Romance",
    "slice of life": "Slice of Life", "horror": "Horror", "mystery": "Mystery", "psychological": "Psychological",
    "supernatural": "Supernatural", "thriller": "Thriller", "isekai": "Isekai", "mecha": "Mecha",
    "mahou shoujo": "Mahou Shoujo", "magical girl": "Mahou Shoujo", "spokon": "Spokon", "sports": "Spokon",
    "iyashikei": "Iyashikei", "healing": "Iyashikei", "harem": "Harem", "reverse harem": "Reverse Harem",
    "ecchi": "Ecchi", "boys love": "Boys' Love", "bl": "Boys' Love", "yaoi": "Boys' Love",
    "girls love": "Girls' Love", "gl": "Girls' Love", "yuri": "Girls' Love", "cgdct": "CGDCT",
    "cute girls doing cute things": "CGDCT", "idol": "Idol / Music", "music": "Idol / Music", 
    "cyberpunk": "Cyberpunk", "steampunk": "Steampunk", "post-apocalyptic": "Post-Apocalyptic", 
    "historical": "Historical", "jidaigeki": "Historical", "dark fantasy": "Dark Fantasy", 
    "military": "Military", "space opera": "Space Opera", "yokai": "Yokai", "youkai": "Yokai",
    "high-stakes game": "High-Stakes Game", "game": "High-Stakes Game", "martial arts": "Martial Arts",
    "tensei": "Tensei", "reincarnation": "Tensei", "villainess": "Villainess", "otome game": "Villainess",
    "school life": "School Life", "school": "School Life", "workplace": "Workplace", 
    "gourmet": "Gourmet / Cooking", "cooking": "Gourmet / Cooking", "food": "Gourmet / Cooking", 
    "delinquent": "Delinquent", "yankee": "Delinquent", "parody": "Parody / Gag", "gag": "Parody / Gag", 
    "time travel": "Time Travel", "loop": "Time Travel", "magic academy": "Magic Academy", 
    "card game": "Card/Board Game", "board game": "Card/Board Game"
}

# Function to handle local background image and custom CSS
def set_bg_from_local(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpg;base64,{encoded_string});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .block-container {{
                background-color: rgba(255, 255, 255, 0.85);
                border-radius: 15px;
                padding: 3rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
                margin-top: 2rem;
            }}
            h1, h2, h3, p, span, .stMarkdown {{
                color: #1A1A1A !important;
            }}
            /* Overriding selectbox styles */
            div[data-baseweb="select"] > div {{
                background-color: #FFFFFF !important;
                border: 2px solid #000000 !important;
                border-radius: 8px !important;
            }}
            div[data-baseweb="select"] * {{
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important; 
                font-weight: 500 !important;
            }}
            ul[data-baseweb="menu"] {{
                background-color: #FFFFFF !important;
                border: 2px solid #000000 !important;
                border-radius: 8px !important;
            }}
            li[data-baseweb="option"] {{
                color: #000000 !important;
            }}
            .stButton > button * {{
                color: white !important;
                -webkit-text-fill-color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"Couldn't load background image at {image_path}")

set_bg_from_local(r"C:\Users\ujjwa\Desktop\project\IITG-P2-main\src\BG_Wallpaper.jpg")

# Core engine initialization with caching
@st.cache_resource
def initialize_engine(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data['data'])
    
    # Strip down to only the columns we actually need
    cols_to_keep = ['title', 'status', 'picture', 'tags', 'episodes', 'score', 'synonyms']
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df = df[df['status'].isin(["FINISHED", "ONGOING"])]
    df = df.dropna(subset=['title'])
    df = df.reset_index(drop=True)
    
    search_mapping = {}
    
    # Add pure titles first
    for title in df['title']:
        clean_title = str(title).strip()
        search_mapping[clean_title] = clean_title
        
    # Process synonyms faster using itertuples
    if 'synonyms' in df.columns:
        for row in df.itertuples():
            main_title = str(row.title).strip()
            syns = getattr(row, 'synonyms', [])
            if isinstance(syns, list):
                for syn in syns:
                    clean_syn = str(syn).strip()
                    # Only add clean english text under 50 chars to avoid lag
                    if clean_syn and clean_syn.isascii() and len(clean_syn) < 50:
                        if clean_syn not in search_mapping:
                            search_mapping[clean_syn] = main_title

    all_dropdown_names = sorted(list(search_mapping.keys()))

    # Process tags
    df['tags'] = df['tags'].apply(lambda x: x if isinstance(x, list) else [])
    df['combined_tags'] = df['tags'].apply(lambda tags: " ".join([str(tag).replace(" ", "") for tag in tags]))
    
    # Formatting the display genres
    def filter_display_genres(raw_tags):
        matched = set()
        for tag in raw_tags:
            clean_tag = str(tag).lower().strip()
            if clean_tag in GENRE_MAPPING:
                matched.add(GENRE_MAPPING[clean_tag])
            else:
                for word in clean_tag.replace("-", " ").split():
                    if word in GENRE_MAPPING:
                        matched.add(GENRE_MAPPING[word])
        if not matched:
            return "Uncategorized"
        genre_list = sorted(list(matched))
        return ", ".join(genre_list[:3]) + ("..." if len(genre_list) > 3 else "")

    df['display_genres'] = df['tags'].apply(filter_display_genres)
    
    # Clean up episodes
    if 'episodes' in df.columns:
        df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').fillna(0).astype(int).astype(str).replace('0', 'Unknown')
    else:
        df['episodes'] = 'Unknown'
        
    def extract_score(score_data):
        if isinstance(score_data, dict) and 'arithmeticMean' in score_data:
            try:
                return f"{float(score_data['arithmeticMean']):.2f}"
            except ValueError:
                pass
        return "N/A"
        
    if 'score' in df.columns:
        df['score'] = df['score'].apply(extract_score)
    else:
        df['score'] = 'N/A'
    
    # Build ML vectors, but skip full matrix calculation here to save RAM
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['combined_tags'])
    
    # Clean up unhashable items
    df = df.drop(columns=['tags', 'combined_tags', 'synonyms'], errors='ignore')
    for col in ['title', 'status', 'picture', 'display_genres', 'episodes', 'score']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            
    return df, tfidf_matrix, all_dropdown_names, search_mapping

# Load everything up
try:
    df, tfidf_matrix, all_dropdown_names, search_mapping = initialize_engine('anime-offline-database-minified.json')
except FileNotFoundError:
    st.error("Make sure 'anime-offline-database-minified.json' is in the right directory.")
    st.stop()

# Initialize session state
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'target_anime' not in st.session_state:
    st.session_state.target_anime = df['title'].iloc[0]

st.title("🎬 Anime Recommendation System")

if not st.session_state.show_results:
    left_spacer, center_col, right_spacer = st.columns([1, 2, 1])
    with center_col:
        st.write("Select an anime you love, and we will find mathematically similar recommendations based on its tags and genres!")
        
        selected_name = st.selectbox("Search Anime (Title or English Alias):", all_dropdown_names, key="init_select")
        
        if st.button("Get Recommendations", type="primary"):
            st.session_state.target_anime = search_mapping[selected_name]
            st.session_state.show_results = True
            st.rerun()
else:
    search_col, results_col = st.columns([1, 2.5])
    
    with search_col:
        st.write("### Search Again")
        
        default_index = all_dropdown_names.index(st.session_state.target_anime) if st.session_state.target_anime in all_dropdown_names else 0
        selected_name = st.selectbox("Search Anime (Title or English Alias):", all_dropdown_names, index=default_index, key="active_select")
        
        if st.button("Get Recommendations", type="primary"):
            st.session_state.target_anime = search_mapping[selected_name]
            st.rerun()
            
    with results_col:
        st.write(f"### If you liked **{st.session_state.target_anime}**, you might also enjoy:")
        st.write("---")
        
        try:
            idx = df.index[df['title'] == st.session_state.target_anime].tolist()[0]
            
            # Calculate similarity dynamically right when needed
            sim_array = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
            sim_scores = list(enumerate(sim_array))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_matches = sim_scores[1:6]
            
            res_cols = st.columns(5)
            for rank, (i, sim_val) in enumerate(top_matches):
                title = df['title'].iloc[i]
                status = df['status'].iloc[i]
                picture_url = df['picture'].iloc[i]
                display_genres = df['display_genres'].iloc[i]
                episodes = df['episodes'].iloc[i]
                score = df['score'].iloc[i]
                
                match_percentage = round(sim_val * 100, 1)
                
                with res_cols[rank]:
                    if picture_url and picture_url.lower() != "nan":
                        img_element = f'<img src="{picture_url}" style="width: 100%; height: 280px; object-fit: cover; border-radius: 10px 10px 0 0;">'
                    else:
                        img_element = f'<div style="width: 100%; height: 280px; background-color: #E0E0E0; border-radius: 10px 10px 0 0; display: flex; align-items: center; justify-content: center; color: #555;">No Image</div>'
                    
                    # HTML card with line-clamping applied to the title
                    card_html = f"""
                    <div style="
                        background-color: rgba(255, 255, 255, 0.95);
                        border-radius: 10px;
                        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
                        display: flex;
                        flex-direction: column;
                        height: 560px; 
                        overflow: hidden;
                        transition: transform 0.2s ease-in-out;
                    ">
                        {img_element}
                        <div style="padding: 12px; display: flex; flex-direction: column; flex-grow: 1;">
                            <h4 style="
                                margin: 0 0 10px 0; 
                                color: #111; 
                                font-size: 1rem; 
                                line-height: 1.3;
                                display: -webkit-box;
                                -webkit-line-clamp: 2;
                                -webkit-box-orient: vertical;
                                overflow: hidden;
                            ">{title}</h4>
                            <div style="margin-top: auto;">
                                <p style="margin: 4px 0; color: #444; font-size: 0.85rem;">🔥 <strong>{match_percentage}% Match</strong></p>
                                <p style="margin: 4px 0; color: #444; font-size: 0.85rem;"><strong>Genre:</strong> {display_genres}</p>
                                <p style="margin: 4px 0; color: #444; font-size: 0.85rem;">📺 Episodes: {episodes}</p>
                                <p style="margin: 4px 0; color: #444; font-size: 0.85rem;">⭐ Score: {score}</p>
                                <p style="margin: 4px 0; color: #444; font-size: 0.85rem;">⚡ Status: {status}</p>
                            </div>
                        </div>
                    </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
        except IndexError:
            st.error("Looks like something went wrong. The selected anime might be missing some critical tags in the database.")
