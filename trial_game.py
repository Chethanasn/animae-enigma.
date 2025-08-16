# ---------------------------------------------------------
# Anime Enigma (Streamlit) - Full Code with Dynamic Background
# ---------------------------------------------------------

import streamlit as st
from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import os
import random
import time
from streamlit_toggle import st_toggle_switch

# ---------- Config ----------
DATASET_ROOT = r"C:\Users\HP\Desktop\gaming\animae_dataset"
MODEL_PATH = "trained_model/anime_classifier_best_model.h5"

MAX_SCORE = 100
INITIAL_BLUR = 15
BLUR_STEP = 5
HINT_COST = 30
WRONG_GUESS_PENALTY = 10

GRID_SIZE = 4
TILE_SIZE = 80
PUZZLE_TIME_LIMIT = 180

st.set_page_config(page_title="Anime Enigma", layout="wide")

# ---------- Theme Toggle ----------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

theme_toggle = st_toggle_switch(
    label="🌗 Toggle Dark Mode",
    key="theme_switch",
    default_value=(st.session_state.theme_mode == "Dark"),
    label_after=True,
    inactive_color="#d3d3d3",
    active_color="#11567f",
    track_color="#29b5e8"
)

st.session_state.theme_mode = "Dark" if theme_toggle else "Light"

if st.session_state.theme_mode == "Dark":
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #0e1117;
            color: #f1f1f1;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        </style>
    """, unsafe_allow_html=True)

# ---------- Dynamic Background Function ----------
def set_dynamic_background(anime_name):
    background_urls = {
        "one_piece": "https://wallpaperaccess.com/full/326869.jpg",
        "naruto": "https://wallpaperaccess.com/full/728282.jpg",
        "attack_on_titan": "https://wallpaperaccess.com/full/8428.jpg",
        "demon_slayer": "https://wallpaperaccess.com/full/4205437.jpg",
        "jujutsu_kaisen": "https://wallpaperaccess.com/full/5868789.jpg"
    }
    bg_url = background_urls.get(anime_name, "")
    if bg_url:
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url('{bg_url}');
                background-size: cover;
                background-attachment: fixed;
                background-position: center;
            }}
            </style>
        """, unsafe_allow_html=True)

# ---------- Model Load ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ---------- Character Dictionary (shortened for example) ----------
character_to_anime = {
    "naruto": "naruto",
    "luffy": "one_piece",
    "eren": "attack_on_titan",
    "tanjiro": "demon_slayer",
    "gojo satoru": "jujutsu_kaisen"
}

character_hints = {
    "naruto": "He dreams of becoming Hokage!",
    "luffy": "He wears a straw hat and wants to be the Pirate King!",
    "eren": "He transforms into a Titan.",
    "tanjiro": "A demon slayer with a kind heart.",
    "gojo satoru": "The strongest sorcerer in Jujutsu Kaisen."
}

# ---------- Helper Functions ----------
def get_random_image():
    characters = list(character_to_anime.keys())
    character = random.choice(characters)
    anime = character_to_anime[character]
    folder = os.path.join(DATASET_ROOT, anime)
    if not os.path.exists(folder):
        return None, None
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('jpg', 'png'))]
    if not images:
        return None, None
    return random.choice(images), character

def blur_image(path, blur_level):
    img = Image.open(path)
    return img.filter(ImageFilter.GaussianBlur(blur_level))

def preprocess_image(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = img[..., :3] if img.shape[-1] == 4 else img
    return np.expand_dims(img, axis=0)

def predict(img):
    labels = ["attack_on_titan", "demon_slayer", "jujutsu_kaisen", "naruto", "one_piece"]
    pred = model.predict(preprocess_image(img))
    return labels[np.argmax(pred)]

# ---------- Game UI ----------
def main():
    st.markdown("""
        <div style='text-align:center; font-size:40px; color:#ff3399;'>🎴 Anime Enigma</div>
    """, unsafe_allow_html=True)

    if "score" not in st.session_state:
        st.session_state.score = MAX_SCORE
        st.session_state.used_hint = False
        st.session_state.character = ""
        st.session_state.image_path = ""
        st.session_state.level_complete = False
        st.session_state.blur = INITIAL_BLUR

    if not st.session_state.character:
        path, char = get_random_image()
        if path and char:
            st.session_state.image_path = path
            st.session_state.character = char
        else:
            st.error("No image found in dataset.")
            return

    anime_name = character_to_anime.get(st.session_state.character, "")
    set_dynamic_background(anime_name)

    st.image(blur_image(st.session_state.image_path, st.session_state.blur), width=300)
    guess = st.text_input("Guess the character")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Hint"):
            if not st.session_state.used_hint and st.session_state.score >= HINT_COST:
                st.session_state.score -= HINT_COST
                st.session_state.blur = max(0, st.session_state.blur - BLUR_STEP)
                st.session_state.used_hint = True
                st.info(character_hints.get(st.session_state.character, "No hint available."))
            else:
                st.warning("Not enough score or hint already used.")
    with col2:
        if st.button("Submit"):
            if guess.strip().lower() == st.session_state.character:
                st.success(f"Correct! The character is {st.session_state.character.title()} 🎉")
                st.session_state.level_complete = True
            else:
                st.session_state.score -= WRONG_GUESS_PENALTY
                st.warning(f"Wrong! Try again. Score: {st.session_state.score}")
                st.session_state.blur = max(0, st.session_state.blur - BLUR_STEP)

    st.markdown(f"### 🔥 Score: {st.session_state.score}")

    if st.session_state.level_complete:
        st.balloons()
        st.image(st.session_state.image_path, caption="Original Image", width=300)
        st.markdown("---")
        st.markdown(f"**Character:** {st.session_state.character.title()}")
        st.markdown(f"**Anime:** {anime_name.replace('_', ' ').title()}")
        st.markdown(f"**AI Predicted:** {predict(Image.open(st.session_state.image_path)).replace('_', ' ').title()}")
        if st.button("Play Again"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

# ---------- Run ----------
if __name__ == "__main__":
    main()
