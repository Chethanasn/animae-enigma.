import streamlit as st
from PIL import Image, ImageFilter
import tensorflow as tf
import numpy as np
import os
import random
import time

# --- Global Configurations ---
# IMPORTANT: Ensure this path is correct for your system!
# Replace with the actual path to your 'animae_dataset' folder
DATASET_ROOT = r"C:\Users\HP\Desktop\gaming\animae_dataset"
MODEL_PATH = "trained_model/anime_classifier_best_model.h5"

# Level 1 specific game settings
MAX_SCORE = 100
INITIAL_BLUR = 15
BLUR_STEP = 5
HINT_COST = 30
WRONG_GUESS_PENALTY = 10

# Level 2 specific puzzle settings
GRID_SIZE = 4
TILE_SIZE = 80 # <<<--- FURTHER REDUCED TILE_SIZE HERE for a smaller puzzle
PUZZLE_TIME_LIMIT = 180 # seconds for the puzzle

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Anime Puzzle Game - AI Guess", layout="centered")
from streamlit_toggle import st_toggle_switch

# --- Theme Mode Setup ---
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"  # Default theme

# --- Toggle Switch UI ---
theme_toggle = st_toggle_switch(
    label="🌗 Toggle Dark Mode",
    key="theme_switch",
    default_value=(st.session_state.theme_mode == "Dark"),
    label_after=True,
    inactive_color="#d3d3d3",
    active_color="#11567f",
    track_color="#29b5e8"
)

# Update theme_mode in session state based on toggle
st.session_state.theme_mode = "Dark" if theme_toggle else "Light"

# --- Apply theme style ---
if st.session_state.theme_mode == "Dark":
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #0e1117;
            color: #f1f1f1;
        }
        .stButton>button, .stTextInput>div>div>input {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        /* Specific for horizontal rule in dark mode if needed, though default usually works */
        hr {
            background-color: #333333; /* Darker grey for dark mode lines */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.success("🌙 Dark Mode Activated")
else:
    st.markdown(
        """
        <style>
        body, .stApp {
            background-color: #ffffff;
            color: #000000;
        }
        .stButton>button, .stTextInput>div>div>input {
            background-color: #f0f0f0;
            color: #000000;
        }
        .stAlert {
            background-color: #e1f5fe !important;
            color: #000000 !important;
            /* --- START OF THE ONLY CHANGES TO ADDRESS THE GAP IN LIGHT MODE --- */
            margin-bottom: 0px !important; /* Reduce bottom margin of alert box */
            padding-bottom: 5px !important; /* Slightly reduce padding if needed */
        }
        .stAlert > div {
            color: #000000 !important;
        }
        /* Make the horizontal rule black in light mode */
        hr {
            background-color: #000000; /* Black color for the line */
            border: none; /* Remove default border that might interfere */
            height: 1px; /* Ensure it has a visible height */
            margin-top: 0px !important; /* Reduce top margin of the horizontal rule */
            margin-bottom: 0px !important; /* Ensure no bottom margin on the horizontal rule either */
        }
        h1 { /* Target the title immediately after the hr */
            margin-top: 0px !important; /* Remove top margin of the h1 heading */
            padding-top: 5px !important; /* Add a small padding if you want a tiny space */
        }
        /* --- END OF THE ONLY CHANGES TO ADDRESS THE GAP IN LIGHT MODE --- */
        </style>
        """,
        unsafe_allow_html=True
    )
    st.info("☀️ Light Mode Active")


# --- Load Model ---
@st.cache_resource
def load_ml_model():
    """Loads the pre-trained Keras model for anime classification."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found at: {MODEL_PATH}. Please check the path.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. This might be due to a TensorFlow version mismatch. Try pip install tensorflow==2.15.0.")
        return None

model = load_ml_model()
if model is None:
    st.stop() # Stop the app if the model cannot be loaded

# --- Character to Anime Mapping & Hints ---
# This dictionary is crucial for the game's logic and hints.
character_to_anime = {
    "luffy": "one_piece", "zoro": "one_piece", "nami":"one_piece",
    "chopper":"one_piece", "sanji":"one_piece", "usopp":"one_piece",
    "vivi":"one_piece", "shanks":"one_piece", "ace":"one_piece",
    "arlong":"one_piece", "buggy":"one_piece", "crocodile":"one_piece",
    "doflamingo":"one_piece", "goldroger":"one_piece", "jimbei":"one_piece",
    "kaido":"one_piece", "robin":"one_piece", "smoker":"one_piece",
    "zeff":"one_piece", "brook":"one_piece",
    "gaara": "naruto", "itachi": "naruto", "kakashi": "naruto",
    "madara": "naruto", "minato uzumaki": "naruto", "naruto": "naruto",
    "obito": "naruto", "pain": "naruto", "sakura": "naruto",
    "shikamaru": "naruto",
    "eren": "attack_on_titan", "mikasa": "attack_on_titan", "annie": "attack_on_titan",
    "armin": "attack_on_titan", "armoured titan": "attack_on_titan", "beast titan": "attack_on_titan",
    "bertolt": "attack_on_titan", "car titan": "attack_on_titan", "colossal titan": "attack_on_titan",
    "connie": "attack_on_titan", "eren titan": "attack_on_titan", "erwin smith": "attack_on_titan",
    "female titan": "attack_on_titan", "founder titan": "attack_on_titan", "grisha": "attack_on_titan",
    "hange": "attack_on_titan", "historia reiss": "attack_on_titan", "jaw titan": "attack_on_titan",
    "levi ackerman": "attack_on_titan", "reiner": "attack_on_titan", "sasha": "attack_on_titan",
    "war hammer titan": "attack_on_titan", "ymir fritz": "attack_on_titan",
    "tanjiro": "demon_slayer", "nezuko": "demon_slayer", "akaza":"demon_slayer",
    "genya":"demon_slayer", "giyu tomioka":"demon_slon_titan", "gyomei himejima":"demon_slayer",
    "inosuke":"demon_slayer", "kagaya ubuyashiki":"demon_slayer", "makomo":"demon_slayer",
    "mitsuri kanroji":"demon_slayer", "muichiro tokito":"demon_slayer", "muzan":"demon_slayer",
    "obanai iguro":"demon_slayer", "rengoku":"demon_slayer", "sabito":"demon_slayer",
    "sankoji urokodaki":"demon_slayer", "sanemi shinazugawa":"demon_slayer",
    "shinobu kocho":"demon_slayer", "tamoyo":"demon_slayer", "tengen uzui":"demon_slayer",
    "zenitsu":"demon_slayer",
    "gojo satoru": "jujutsu_kaisen", "itadori yuju":"jujutsu_kaisen",
    "geto suguru":"jujutsu_kaisen", "mahito":"jujutsu_kaisen", "maki zenin":"jujutsu_kaisen",
    "megumi":"jujutsu_kaisen", "nanami":"jujutsu_kaisen", "nobara":"jujutsu_kaisen",
    "sukuna":"jujutsu_kaisen", "toji":"jujutsu_kaisen", "yuta":"jujutsu_kaisen",
}

character_hints = {
    "luffy": "He wears a straw hat and wants to be the Pirate King!",
    "zoro": "A swordsman from the Straw Hat crew.",
    "ace":"He is the fiery older brother of Luffy, known for his flame-based powers.",
    "nami":"She’s the clever navigator of the Straw Hat crew, with a deep love for treasure and maps.",
    "sanji":"A master chef with deadly kicks, he never lets a lady go hungry.",
    "chopper":"A blue-nosed reindeer who’s both a doctor and a cutie — don’t let his cuddly form fool you!",
    "usopp":"A sniper with a long nose and even longer tales — brave when it counts, even if he’s scared stiff!",
    "vivi":"A princess who once sailed undercover with the Straw Hats, fighting to save her desert kingdom from chaos.",
    "crocodile":"A former Warlord with a hook for a hand and mastery over sand, he once sought to overthrow a desert kingdom.",
    "goldroger":"The Pirate King who conquered the Grand Line and left behind the ultimate treasure that started the Great Pirate Era.",
    "shanks":"A legendary red-haired pirate and one of the Four Emperors, known for inspiring Luffy to become a pirate.",
    "arlong":"A ruthless fish-man and former member of the Sun Pirates, he terrorized Nami’s village with dreams of species superiority.",
    "buggy":"This clown-faced pirate has the power to split his body into pieces and once sailed under the same flag as the Pirate King.",
    "doflamingo":"A former Warlord of the Sea known for his flamboyant style and deadly thread-controlling powers.",
    "jimbei":"A powerful fish-man and skilled helmsman, known as the ‘Knight of the Sea’ and a former Warlord of the Sea.",
    "kaido":"One of the Four Emperors, known as the 'Strongest Creature in the World,' with the ability to transform into a massive dragon.",
    "robin":"Archaeologist of the Straw Hat crew who can sprout extra limbs anywhere using the power of the Flower-Flower Fruit.",
    "smoker":"A Marine officer who wields the power of the Smoke-Smoke Fruit, using smoke-based attacks to chase down pirates.",
    "zeff":"The fiery chef and former pirate captain who runs the Baratie floating restaurant and taught Sanji how to cook and fight.",
    "brook":"The musician of the Straw Hat Pirates who came back to life thanks to a mysterious fruit and carries a cane sword.",
    "gaara": "The jinchuriki of the One-Tail beast, known for his sand manipulation and initially feared by many.",
    "itachi": "A prodigy of the Uchiha clan, known for his Sharingan and complex motives behind his actions.",
    "kakashi": "The Copy Ninja, famous for his Sharingan eye and teaching Team 7.",
    "madara": "Legendary Uchiha clan leader and one of the founders of the Hidden Leaf Village.",
    "minato uzumaki":"Known as the 'Yellow Flash,' he is the Fourth Hokage and Naruto’s father.",
    "obito":"Once thought dead, he became the masked villain behind the Akatsuki’s plans.",
    "pain":"Leader of the Akatsuki, he controls six bodies and seeks peace through pain.",
    "sakura":"A skilled medical ninja with incredible strength and a key member of Team 7.",
    "shikamaru":"A genius strategist known for his shadow manipulation and laid-back attitude.",
    "naruto": "He dreams of becoming Hokage!",
    "eren": "He transforms into a Titan.",
    "mikasa":"A fiercely loyal soldier skilled in vertical maneuvering and one of the strongest fighters in her squad.",
    "annie":"A mysterious warrior known for her exceptional combat skills and the power to transform into the Female Titan.",
    "armin":"A brilliant strategist with a kind heart, known for his intelligence and tactical thinking in battles against the Titans.",
    "armoured titan":"A Titan with hardened, armor-like skin, known for its incredible defense and powerful charges during battles.",
    "beast titan":"A Titan with apelike features, known for its intelligence and ability to throw objects with deadly precision.",
    "bertolt":"A quiet and reserved soldier who holds a secret — he can transform into the towering Colossal Titan.",
    "car titan":"A unique Titan form known for its incredible speed and quadrupedal, beast-like appearance, used in fast attacks and transportation.",
    "colossal titan":"A towering Titan famous for its enormous size and the ability to emit explosive steam, causing massive destruction.",
    "connie":"A brave and loyal member of the Survey Corps known for his energetic personality and close friendship with Sasha.",
    "eren titan":"The protagonist who possesses the power to transform into a powerful Titan and fights to protect humanity from the Titans.",
    "erwin smith":"The courageous and strategic commander of the Survey Corps, known for his leadership and unwavering dedication to uncovering the truth beyond the walls.",
    "female titan":"A mysterious and deadly Titan known for agility and combat skills, with the ability to harden parts of her body, often hunting the Survey Corps.",
    "founder titan":"The rare and powerful Titan that holds the key to controlling other Titans and the history of Eldians, passed down through the royal bloodline.",
    "grisha":"A doctor with a mysterious past who holds secrets that change the fate of the world and is the father of the protagonist.",
    "hange":"An eccentric and passionate scientist obsessed with studying Titans, known for their curious and energetic personality.",
    "historia reiss":"The true heir to the royal family, known for her gentle nature and hidden strength.",
    "jaw titan":"A swift and powerful titan known for its incredible jaw strength and biting power.",
    "levi ackerman":"Humanity's strongest soldier, known for his unmatched agility and precision with dual blades.",
    "reiner":"A warrior who can transform into the powerful Armored Titan, known for his strong defense and conflicted loyalty.",
    "sasha":"Known as the ‘Potato Girl,’ she’s a skilled hunter with an insatiable love for food and a big heart.",
    "war hammer titan":"A unique Titan with the power to create weapons and structures from hardened crystal-like material, controlled remotely from a hidden crystal.",
    "ymir fritz":"The original Titan ancestor, whose mysterious power started the Titan lineage over a thousand years ago.",
    "tanjiro": "He is a kind-hearted boy who joins the Demon Slayer Corps to avenge his family and save his sister.",
    "nezuko": "She is a demon, but retains her humanity and often travels in a wooden box on her brother's back.",
    "akaza":"A powerful Upper Rank Three demon with a deep-seated desire to become stronger.",
    "genya":"A hot-headed member of the Demon Slayer Corps who eats demons to gain their powers.",
    "giyu tomioka":"The quiet and reserved Water Hashira.",
    "gyomei himejima":"The blind Stone Hashira, known for his immense strength and gentle nature.",
    "inosuke":"A wild boy raised by boars, who wears a boar's head and uses two jagged swords.",
    "kagaya ubuyashiki":"The benevolent leader of the Demon Slayer Corps.",
    "makomo":"A kind and gentle spirit who helps train Tanjiro.",
    "mitsuri kanroji":"The loving and energetic Love Hashira.",
    "muichiro tokito":"The forgetful but incredibly skilled Mist Hashira.",
    "muzan":"The main antagonist, the first demon and progenitor of all other demons.",
    "obanai iguro":"The strict and snake-loving Serpent Hashira.",
    "rengoku":"The flamboyant Flame Hashira, known for his strong will and powerful attacks.",
    "sabito":"A skilled swordsman who trained under Sakonji Urokodaki.",
    "sankoji urokodaki":"Tanjiro's first trainer, a former Water Hashira.",
    "sanemi shinazugawa":"The volatile Wind Hashira.",
    "shinobu kocho":"The graceful and quick Insect Hashira.",
    "tamoyo":"A compassionate demon doctor who helps humans and seeks to defeat Muzan.",
    "tengen uzui":"The flashy Sound Hashira.",
    "zenitsu":"A cowardly but powerful demon slayer who excels when he's asleep.",
    "gojo satoru": "jujutsu_kaisen", "itadori yuju":"jujutsu_kaisen",
    "geto suguru":"jujutsu_kaisen", "mahito":"jujutsu_kaisen", "maki zenin":"jujutsu_kaisen",
    "megumi":"jujutsu_kaisen", "nanami":"jujutsu_kaisen", "nobara":"jujutsu_kaisen",
    "sukuna":"jujutsu_kaisen", "toji":"jujutsu_kaisen", "yuta":"jujutsu_kaisen",
}

# --- Helper Functions (General for Model and Image Processing) ---
@st.cache_resource
def get_random_image_and_character():
    """
    Randomly selects an image and infers the character name based on
    the dataset structure (subfolders or filenames).
    """
    anime_dirs = [
        os.path.join(DATASET_ROOT, "attack_on_titan"),
        os.path.join(DATASET_ROOT, "demon_slayer"),
        os.path.join(DATASET_ROOT, "jujutsu_kaisen"),
        os.path.join(DATASET_ROOT, "naruto"),
        os.path.join(DATASET_ROOT, "one_piece")
    ]

    all_image_paths_and_chars = []
    for dir_path in anime_dirs:
        if not os.path.exists(dir_path):
            st.warning(f"Dataset directory not found: {dir_path}. Skipping.")
            continue

        for root, _, files in os.walk(dir_path):
            for filename in files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, filename)
                    found_char = None

                    # 1. Try to infer character from parent folder name first (e.g., 'naruto/naruto_uzumaki/image.jpg' -> 'naruto uzumaki')
                    parent_folder_name = os.path.basename(root).lower().replace('_', ' ')
                    if parent_folder_name in character_to_anime:
                        found_char = parent_folder_name
                    else:
                        # 2. Try to infer from filename directly (e.g., 'giyu_tomioka.png' -> 'giyu tomioka')
                        char_candidate_from_filename = os.path.splitext(filename)[0].lower().replace('_', ' ')
                        if char_candidate_from_filename in character_to_anime:
                            found_char = char_candidate_from_filename
                        else:
                            # 3. More flexible matching for multi-word names or partial matches in filename
                            # Iterate through character_to_anime keys to find a match in the filename
                            for char_key_in_dict in character_to_anime.keys():
                                # Check if the filename contains the character key (e.g., 'levi ackerman.jpg' contains 'levi ackerman')
                                # Or if the first word of the filename matches the first word of the character key
                                # Added a check for minimum length to avoid matching very short, generic words
                                if char_key_in_dict in char_candidate_from_filename or \
                                    (char_key_in_dict.split(" ")[0] == char_candidate_from_filename.split(" ")[0] and \
                                     len(char_key_in_dict.split(" ")[0]) >= 3):
                                    found_char = char_key_in_dict # Assign the exact key from the dictionary
                                    break # Stop after finding the first suitable match

                    if found_char:
                        all_image_paths_and_chars.append((full_path, found_char))

    if not all_image_paths_and_chars:
        st.error("No images with recognized characters found in the specified dataset directories. Please ensure paths are correct and names align with your 'character_to_anime' mapping.")
        return None, None

    return random.choice(all_image_paths_and_chars)

def get_blurred_image(path, blur_level):
    """Applies a Gaussian blur to an image."""
    if path is None or not os.path.exists(path):
        return Image.new('RGB', (300, 300), color='grey') # Placeholder if image not found
    img = Image.open(path)
    return img.filter(ImageFilter.GaussianBlur(blur_level))

def preprocess_image(img, target_size=(224, 224)):
    """Preprocesses an image for model prediction."""
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    if img_array.shape[-1] == 4: # Handle images with an alpha channel
        img_array = img_array[..., :3]
    return np.expand_dims(img_array, axis=0)

def predict_label(img):
    """Predicts the anime label for a given image using the loaded model."""
    preprocessed = preprocess_image(img)
    preds = model.predict(preprocessed)
    class_idx = np.argmax(preds)
    label_map = {
        0: "attack_on_titan",
        1: "demon_slayer",
        2: "jujutsu_kaisen",
        3: "naruto",
        4: "one_piece"
    }
    return label_map.get(class_idx, "unknown")

# --- Level 2 Puzzle Specific Helper Functions ---
def slice_image(image_path, grid_size=GRID_SIZE):
    """Slices an image into a grid of tiles."""
    if not os.path.exists(image_path):
        st.error(f"Puzzle image not found at: {image_path}")
        # Return a grid of solid red tiles if image is missing
        return [Image.new('RGB', (TILE_SIZE, TILE_SIZE), color='red')] * (grid_size * grid_size)

    img = Image.open(image_path)
    # Ensure image is square and resizes to fit tile grid perfectly
    img = img.resize((TILE_SIZE * grid_size, TILE_SIZE * grid_size))
    tiles = []
    for i in range(grid_size):
        for j in range(GRID_SIZE):
            box = (j * TILE_SIZE, i * TILE_SIZE, (j + 1) * TILE_SIZE, (i + 1) * TILE_SIZE)
            tile = img.crop(box)
            tiles.append(tile)
    return tiles

def shuffle_tiles(tiles):
    """Shuffles the order of tiles and returns the shuffled tiles and their new indices."""
    indices = list(range(len(tiles)))
    random.shuffle(indices)
    shuffled = [tiles[i] for i in indices]
    return shuffled, indices

def is_puzzle_solved(shuffled_tiles_list, original_tiles_list):
    """Checks if the current arrangement of tiles matches the original order."""
    if len(shuffled_tiles_list) != len(original_tiles_list):
        return False
    # Compare PIL Image objects directly (checks if they are the same object, which works for swaps)
    for i in range(len(shuffled_tiles_list)):
        if shuffled_tiles_list[i] is not original_tiles_list[i]:
            return False
    return True

# --- Game State Management ---
def initialize_level_state(level_num):
    """
    Initializes or resets the game state for a new level.
    This includes fetching a new random image/character and setting up level-specific states.
    """
    st.session_state.current_level = level_num

    if level_num == 1:
        # --- Level 1 Specific Setup (Character Guessing) ---
        path, char_name = get_random_image_and_character()
        if path is None:
            st.stop() # Stop the app if no valid image is found
        st.session_state.random_image_path = path
        st.session_state.current_character_name = char_name
        st.session_state.score = MAX_SCORE
        st.session_state.blur_level = INITIAL_BLUR
        st.session_state.used_hint = False # For L1 hint
        st.session_state.level_completed = False # For L1 completion
        st.session_state.guess_input_value = "" # Clear the text input for L1
        st.session_state.correct_character_image_path = None # Clear path for L2's image
        st.session_state.ai_prediction = "" # Initialize AI prediction for L1
        
        # Initialize new feedback variables for Level 1
        st.session_state.score_feedback_message = ""
        st.session_state.hint_message = ""
        st.session_state.guess_feedback_message = ""

        # --- Clear Level 2 Specific States when transitioning back to Level 1 ---
        st.session_state.level2_started = False
        st.session_state.shuffled_tiles = []
        st.session_state.original_tiles = []
        st.session_state.selected_tile = None
        st.session_state.start_time_l2 = None
        st.session_state.level2_hint_used = False
        st.session_state.level2_puzzle_solved = False

    elif level_num == 2:
        # --- Level 2 Specific Setup (Image Puzzle) ---
        # Ensure we have the image path from Level 1's correct answer
        if 'correct_character_image_path' not in st.session_state or \
           st.session_state.correct_character_image_path is None or \
           not os.path.exists(st.session_state.correct_character_image_path):
            st.error("Error: No valid character image found from Level 1 to start Level 2 puzzle. Returning to Level 1.")
            initialize_level_state(1) # Fallback to Level 1 if image is missing
            return

        # Use the character image from Level 1 for the puzzle
        image_to_puzzle_path = st.session_state.correct_character_image_path
        
        # Prepare tiles for the puzzle
        st.session_state.original_tiles = slice_image(image_to_puzzle_path, GRID_SIZE)
        st.session_state.shuffled_tiles, _ = shuffle_tiles(st.session_state.original_tiles)
        
        # Ensure the puzzle starts shuffled (not already solved by chance)
        attempts = 0
        while is_puzzle_solved(st.session_state.shuffled_tiles, st.session_state.original_tiles) and attempts < 100:
            st.session_state.shuffled_tiles, _ = shuffle_tiles(st.session_state.original_tiles)
            attempts += 1
        if attempts >= 100:
            st.warning("Could not create a shuffled puzzle after many attempts. Displaying as is.")

        st.session_state.level2_started = True
        st.session_state.selected_tile = None # For swapping tiles
        st.session_state.start_time_l2 = time.time() # Start timer for Level 2
        st.session_state.level2_hint_used = False # Reset hint for Level 2
        st.session_state.level2_puzzle_solved = False # Reset solved state for Level 2
        # Level 1's score carries over by default. Uncomment below if you want Level 2 to have its own fresh score:
        # st.session_state.score = MAX_SCORE

# --- Callbacks for UI actions ---
def handle_hint():
    """Handles logic when the 'Need a Hint?' button is clicked for Level 1."""
    st.session_state.score_feedback_message = "" # Clear previous score messages
    st.session_state.hint_message = "" # Clear previous hint messages
    st.session_state.guess_feedback_message = "" # Clear guess feedback too

    if not st.session_state.level_completed: # This applies to Level 1 only
        if st.session_state.score >= HINT_COST:
            if not st.session_state.used_hint:
                st.session_state.score -= HINT_COST
                st.session_state.score_feedback_message = f"Score -{HINT_COST} for using a hint! Current Score: {st.session_state.score}" # Display score change
                st.session_state.used_hint = True
            else:
                st.session_state.score_feedback_message = "You've already used a hint for this level, but the blur will decrease. Current Score: {st.session_state.score}" # Display score

            st.session_state.blur_level = max(st.session_state.blur_level - BLUR_STEP, 0)
            actual_character = st.session_state.current_character_name

            hint_text = character_hints.get(actual_character, "No specific hint available for this character yet. Defaulting to 'He dreams of becoming Hokage!'.")
            if hint_text == "No specific hint available for this character yet. Defaulting to 'He dreams of becoming Hokage!'.":
                # This is a fallback hint if no specific hint is found, or if the name is not in the dictionary.
                # Make it very obvious this is the default.
                hint_text = "He dreams of becoming Hokage! (Default Hint - Character specific hint not found.)"

            st.session_state.hint_message = f"Hint: {hint_text}"
        else:
            st.session_state.score_feedback_message = f"You don't have enough score for a hint! Current Score: {st.session_state.score}" # Display score
    else:
        st.session_state.score_feedback_message = "Level already completed! No more hints needed."


def handle_submit_guess():
    """Handles logic when the 'Submit Guess' button is clicked for Level 1."""
    guess = st.session_state.guess_input_value.strip().lower()
    st.session_state.score_feedback_message = "" # Clear score messages when guessing
    st.session_state.hint_message = "" # Clear hint messages when guessing
    st.session_state.guess_feedback_message = "" # New variable for guess-related feedback

    if st.session_state.level_completed:
        st.session_state.guess_feedback_message = "Level already completed! Click 'Go to Level 2' to proceed."
        return

    if not guess:
        st.session_state.guess_feedback_message = "Please enter a guess."
        return

    st.session_state.guess_input_value = "" # Clear the input field

    if not os.path.exists(st.session_state.random_image_path):
        st.error(f"Image file not found: {st.session_state.random_image_path}. Cannot process guess.")
        return

    original_img = Image.open(st.session_state.random_image_path)
    predicted_anime_by_ai = predict_label(original_img).lower()
    st.session_state.ai_prediction = predicted_anime_by_ai # Store AI prediction for display

    is_correct_character = (guess == st.session_state.current_character_name)

    if is_correct_character:
        st.session_state.level_completed = True
        st.session_state.correct_character_image_path = st.session_state.random_image_path # Store for Level 2
    else:
        if guess in character_to_anime:
            guessed_char_anime = character_to_anime.get(guess).replace('_', ' ').title()
            st.session_state.guess_feedback_message += f"'{guess.title()}' is a character from {guessed_char_anime}, but that's not who this is."
        else:
            st.session_state.guess_feedback_message += f"'{guess.title()}' is not in our character list. Try another name!"

        st.session_state.guess_feedback_message += f"\nAI predicted the anime as: {predicted_anime_by_ai.replace('_', ' ').title()}."

        st.session_state.blur_level = max(st.session_state.blur_level - BLUR_STEP, 0)
        st.session_state.score = max(0, st.session_state.score - WRONG_GUESS_PENALTY)
        st.session_state.guess_feedback_message += f"\nScore -{WRONG_GUESS_PENALTY} for incorrect guess. Current score: {st.session_state.score}"


def go_to_level_2_callback():
    """Callback to advance to Level 2."""
    initialize_level_state(2) # This will reset states specific to Level 2 puzzle and prepare it

def go_to_level_1_callback():
    """Callback to return to Level 1."""
    initialize_level_state(1) # This will reset states specific to Level 1 puzzle and prepare it

# --- Level 1: Guess the Character ---

def run_level_1():
    """Renders the UI and handles logic for Level 1."""
    st.title("🔍 Anime Enigma - Level 1")
    st.markdown("---") # This horizontal rule will be affected by the CSS change

    if st.session_state.random_image_path:
        blurred_img = get_blurred_image(st.session_state.random_image_path, st.session_state.blur_level)
        st.image(blurred_img, caption=f"Guess the character! Current Blur: {st.session_state.blur_level}", width=300)
    else:
        st.warning("No image loaded for the puzzle. Please check dataset configuration.")

    # Removed the line that initially displays "Current Score: 100" here.
    # The score will now be displayed as part of feedback messages after interaction.

    # Use a unique key for this text input to prevent issues on state resets
    st.text_input("Who is in the image? (Enter character name)",
                      key="guess_input_value",
                      value=st.session_state.guess_input_value
                      )

    col1, col2 = st.columns(2)
    with col1:
        st.button("Need a Hint?", on_click=handle_hint, disabled=st.session_state.level_completed)
    with col2:
        st.button("Submit Guess", on_click=handle_submit_guess, disabled=st.session_state.level_completed)

    # --- Display feedback messages here, AFTER the main UI elements ---
    if st.session_state.get("score_feedback_message"):
        st.warning(st.session_state.score_feedback_message)
        # Optional: Clear after displaying if it's a one-time message
        # st.session_state.score_feedback_message = ""

    if st.session_state.get("hint_message"):
        st.info(st.session_state.hint_message)
        # Optional: Clear after displaying if it's a one-time message
        # st.session_state.hint_message = ""

    if st.session_state.get("guess_feedback_message"):
        st.info(st.session_state.guess_feedback_message) # Using info for guess feedback
        # Optional: Clear after displaying
        # st.session_state.guess_feedback_message = ""

    st.markdown("---") # This horizontal rule will be affected by the CSS change

    if st.session_state.level_completed:
        st.success("🎉 Congratulations! You've completed Level 1!")
        
        # Display each result on a separate line
        st.write(f"✅ The character was indeed: {st.session_state.current_character_name.title()}")
        st.write(f"Its anime is: {character_to_anime.get(st.session_state.current_character_name, 'an unknown anime').replace('_', ' ').title()}.")
        st.write(f"📊 Your final score for this level: {st.session_state.score}")
        st.write(f"🤖 The AI predicted the anime as: {st.session_state.ai_prediction.replace('_', ' ').title()}.")

        # Display the "Go to Level 2" button last
        st.button("Go to Level 2", on_click=go_to_level_2_callback)


# --- Level 2: Solve the Puzzle! ---

def run_level_2():
    """Renders the UI and handles logic for Level 2 (Image Puzzle)."""
    st.title("🧩 Anime Enigma - Level 2: Puzzle!")
    st.markdown("---") # This horizontal rule will be affected by the CSS change

    # Use specific Level 2 timer state
    elapsed_time = int(time.time() - st.session_state.start_time_l2)
    remaining_time = PUZZLE_TIME_LIMIT - elapsed_time

    # Display timer and check if time is up
    if remaining_time <= 0 and not st.session_state.level2_puzzle_solved:
        st.error("⏰ Time's up! Puzzle failed.")
        # Offer option to retry Level 2 immediately
        if st.button("Try Level 2 Again", on_click=lambda: initialize_level_state(2), key="l2_retry_fail_btn"):
            pass # Rerun to restart Level 2
    else:
        st.info(f"⏳ Time Left: {remaining_time} seconds")

    # Hint Button for Level 2
    # Only show hint if puzzle is not solved and time is not up
    if not st.session_state.level2_hint_used and not st.session_state.level2_puzzle_solved and remaining_time > 0:
        if st.button("🔍 Show Hint (once for Level 2)", key="l2_hint_btn"):
            if st.session_state.correct_character_image_path and os.path.exists(st.session_state.correct_character_image_path):
                # Made the hint image smaller by setting a fixed width
                st.image(Image.open(st.session_state.correct_character_image_path), caption="Hint: Original Image", width=200) # Adjust width as needed
                st.session_state.level2_hint_used = True
                time.sleep(2)  # Display for 2 seconds
                st.rerun()
            else:
                st.warning("No hint image available for this puzzle.")

    # Puzzle Grid Display
    # Ensure tiles are available before trying to display them
    if 'shuffled_tiles' in st.session_state and st.session_state.shuffled_tiles:
        
        # Display the grid in a structured way using Streamlit columns
        for i in range(GRID_SIZE):
            row_cols = st.columns(GRID_SIZE) # Create a row of columns for each row of the grid
            for j in range(GRID_SIZE):
                idx = i * GRID_SIZE + j
                tile = st.session_state.shuffled_tiles[idx]
                with row_cols[j]:
                    # Create a button for each tile. Label is empty, but key makes it unique.
                    # Disable buttons if puzzle is solved or time is up
                    if st.button(" ", key=f"l2_tile_btn_{idx}", use_container_width=True,
                                 disabled=st.session_state.level2_puzzle_solved or remaining_time <= 0):
                        
                        # Handle tile selection and swapping
                        if st.session_state.selected_tile is None:
                            st.session_state.selected_tile = idx
                            # Removed the previous st.success message here
                        else:
                            # Swap tiles
                            first_idx = st.session_state.selected_tile
                            second_idx = idx
                            
                            # Perform the swap in the shuffled_tiles list
                            temp_tile = st.session_state.shuffled_tiles[first_idx]
                            st.session_state.shuffled_tiles[first_idx] = st.session_state.shuffled_tiles[second_idx]
                            st.session_state.shuffled_tiles[second_idx] = temp_tile
                            
                            st.session_state.selected_tile = None # Reset selected tile

                            # After swap, check if solved
                            if is_puzzle_solved(st.session_state.shuffled_tiles, st.session_state.original_tiles):
                                st.session_state.level2_puzzle_solved = True
                                st.balloons() # Celebrate!
                                st.success(f"🎉 Puzzle Solved! You completed Level 2 in {elapsed_time} seconds!")
                            
                            # Rerun to update the display with swapped tiles
                            st.rerun()
                    
                    # Display the image for the tile
                    st.image(tile, use_container_width=True)


    # Logic for when puzzle is solved or time is up
    if st.session_state.level2_puzzle_solved:
        st.success("You solved Level 2!")
        st.write("Ready for more challenges?")
        st.button("Return to Level 1", on_click=go_to_level_1_callback, key="l2_return_to_l1_solved")
    elif remaining_time <= 0:
        st.error("Time is up! You did not solve the puzzle.")
        st.button("Return to Level 1", on_click=go_to_level_1_callback, key="l2_return_to_l1_timed_out")


# --- Main Application Logic ---
if __name__ == "__main__":
    # Initialize session state if not already done
    if 'current_level' not in st.session_state:
        initialize_level_state(1) # Start at Level 1

    # Route based on current level
    if st.session_state.current_level == 1:
        run_level_1()
    elif st.session_state.current_level == 2:
        run_level_2()