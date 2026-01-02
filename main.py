import streamlit as st
from PIL import Image, ImageFilter
import os, random, time

# ================= CONFIG =================
DATASET_ROOT = r"C:\Users\Admin\Desktop\gaming\animae_dataset"

GRID_SIZE = 4
TILE_SIZE = 90

BLUR_START = 15
BLUR_STEP = 5

START_SCORE = 100
HINT_PENALTY = 20
WRONG_PENALTY = 10

LEVEL2_TIME_LIMIT = 180  # 3 minutes

st.set_page_config("Anime Enigma", layout="centered")

# ================= SESSION INIT =================
if "dark" not in st.session_state:
    st.session_state.dark = False

if "level" not in st.session_state:
    st.session_state.level = 1
    st.session_state.score = START_SCORE
    st.session_state.blur = BLUR_START
    st.session_state.hint_used = False
    st.session_state.correct = False

# ================= THEME TOGGLE =================
st.session_state.dark = st.checkbox("🌙 Dark Mode", value=st.session_state.dark)

if st.session_state.dark:
    st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at top, #1b1f2a, #0b0e14 70%);
        color: #e5e7eb;
    }
    h1,h2,h3,label,p,span { color:#e5e7eb !important; }
    input { background:#111827 !important; color:white !important; }
    .stButton>button {
        background: linear-gradient(135deg,#2563eb,#1e40af);
        color:white !important;
        border-radius:8px;
        width:100%;
    }
    .stAlert-success { background:#064e3b !important; color:#d1fae5 !important; }
    .stAlert-error { background:#7f1d1d !important; color:#fee2e2 !important; }
    .stAlert-info { background:#1e3a8a !important; color:#dbeafe !important; }
    </style>
    """, unsafe_allow_html=True)

# ================= CHARACTER → ANIME =================
character_to_anime = {
    "shanks":"one_piece","luffy":"one_piece","zoro":"one_piece",
    "naruto":"naruto","kakashi":"naruto","itachi":"naruto",
    "eren":"attack_on_titan","mikasa":"attack_on_titan","levi":"attack_on_titan",
    "tanjiro":"demon_slayer","nezuko":"demon_slayer",
    "gojo":"jujutsu_kaisen","sukuna":"jujutsu_kaisen"
}

character_hints = {
    "shanks":"A red-haired pirate who inspired Luffy.",
    "luffy":"Wears a straw hat and wants to be Pirate King.",
    "naruto":"Dreams of becoming Hokage.",
    "eren":"He can transform into a Titan.",
    "tanjiro":"A demon slayer saving his sister.",
    "gojo":"The strongest sorcerer."
}

# ================= LOAD DATA =================
@st.cache_resource
def load_data():
    data = []
    for anime in os.listdir(DATASET_ROOT):
        anime_path = os.path.join(DATASET_ROOT, anime)
        if not os.path.isdir(anime_path):
            continue
        for img in os.listdir(anime_path):
            name = img.lower()
            for char in character_to_anime:
                if char in name:
                    data.append((os.path.join(anime_path, img), char, anime))
    return data

DATA = load_data()
if not DATA:
    st.error("❌ Dataset empty or filename mismatch")
    st.stop()

# ================= NEW ROUND =================
def new_round():
    st.session_state.img, st.session_state.char, st.session_state.anime = random.choice(DATA)
    st.session_state.blur = BLUR_START
    st.session_state.score = START_SCORE
    st.session_state.hint_used = False
    st.session_state.correct = False

if "img" not in st.session_state:
    new_round()

# ================= LEVEL 1 =================
if st.session_state.level == 1:
    st.title("🔍 Level 1: Guess the Character")

    st.metric("Current Score", st.session_state.score)

    img = Image.open(st.session_state.img)
    st.image(img.filter(ImageFilter.GaussianBlur(st.session_state.blur)), width=280)

    guess = st.text_input("Your guess").strip().lower()

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Hint", disabled=st.session_state.hint_used or st.session_state.correct):
            st.session_state.blur = max(0, st.session_state.blur - BLUR_STEP)
            st.session_state.score -= HINT_PENALTY
            st.session_state.hint_used = True
            st.info("💡 " + character_hints.get(st.session_state.char, "Very famous character"))

        if st.session_state.correct:
            st.success(
                f"✅ Correct! {st.session_state.char.title()} "
                f"({st.session_state.anime.replace('_',' ').title()})"
            )
            st.success(f"🏆 Final Score: {st.session_state.score}")

            if st.button("Go to Level 2"):
                st.session_state.level = 2
                st.session_state.level2_start = time.time()

    with col2:
        if st.button("Submit", disabled=st.session_state.correct):
            if guess == st.session_state.char:
                st.session_state.correct = True
            else:
                st.error("❌ Wrong! Try again.")
                st.session_state.blur = max(0, st.session_state.blur - BLUR_STEP)
                st.session_state.score -= WRONG_PENALTY

# ================= LEVEL 2 =================
if st.session_state.level == 2:
    st.title("🧩 Level 2: Puzzle")

    elapsed = int(time.time() - st.session_state.level2_start)
    remaining = LEVEL2_TIME_LIMIT - elapsed

    if remaining <= 0:
        st.error("⏰ Time's up! Level 2 Failed.")
        if st.button("Back to Level 1"):
            st.session_state.level = 1
            new_round()
        st.stop()

    st.info(f"⏳ Time Left: {remaining} seconds")

    # ===== Reference image + Puzzle grid =====
    left, right = st.columns([1, 2])

    base = Image.open(st.session_state.img).resize(
        (GRID_SIZE*TILE_SIZE, GRID_SIZE*TILE_SIZE)
    )

    tiles = [
        base.crop((j*TILE_SIZE, i*TILE_SIZE,
                   (j+1)*TILE_SIZE, (i+1)*TILE_SIZE))
        for i in range(GRID_SIZE) for j in range(GRID_SIZE)
    ]

    if "puzzle" not in st.session_state:
        st.session_state.puzzle = random.sample(tiles, len(tiles))
        st.session_state.sel = None

    with left:
        st.markdown("### 🖼️ Reference Image")
        st.image(base.resize((220, 220)))

    with right:
        for i in range(GRID_SIZE):
            cols = st.columns(GRID_SIZE)
            for j in range(GRID_SIZE):
                idx = i*GRID_SIZE + j
                with cols[j]:
                    if st.button(" ", key=f"tile_{idx}"):
                        if st.session_state.sel is None:
                            st.session_state.sel = idx
                        else:
                            a, b = st.session_state.sel, idx
                            st.session_state.puzzle[a], st.session_state.puzzle[b] = \
                                st.session_state.puzzle[b], st.session_state.puzzle[a]
                            st.session_state.sel = None
                    st.image(st.session_state.puzzle[idx])

    if st.session_state.puzzle == tiles:
        bonus = remaining // 2
        level2_score = 50 + bonus
        st.balloons()
        st.success("🎉 Level 2 Cleared!")
        st.success(f"🏆 Level 2 Score: {level2_score}")

        if st.button("Play Again"):
            st.session_state.level = 1
            new_round()
