from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import unicodedata
import os
from sentence_transformers import SentenceTransformer, util
from Levenshtein import ratio
from rapidfuzz import fuzz
from textblob import TextBlob
from spellchecker import SpellChecker
import torch

# Flask setup
app = Flask(__name__)
CORS(app)

# Load model and spellchecker
os.environ['HUB_DISABLE_SYMLINKS_WARNING'] = '1'
model = SentenceTransformer("paraphrase-MinILM-L6-v2")
spell = SpellChecker()

# Text preprocessing function
def clean_and_correct_text(text):
    text = text.lower().strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = " ".join(text.split())
    
    # Grammar correction with TextBlob
    corrected_text = str(TextBlob(text).correct())

    # Spell checking word-by-word
    corrected_words = []
    for word in text.split():
        if word in spell:
            corrected_words.append(word)
        else:
            correction = spell.correction(word)
            corrected_words.append(correction if correction else word)

    corrected_text = " ".join(corrected_words)
    return corrected_text

# Main feedback evaluation route
@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json()
    user_answer = data.get("user_answer", "")
    correct_answer = data.get("correct_answer", "")

    # Preprocess both answers
    clean_user = clean_and_correct_text(user_answer)
    clean_correct = clean_and_correct_text(correct_answer)

    # Embeddings
    emb_user = model.encode(clean_user, convert_to_tensor=True, normalize_embeddings=True)
    emb_correct = model.encode(clean_correct, convert_to_tensor=True, normalize_embeddings=True)

    # Similarity metrics
    semantic_sim = util.pytorch_cos_sim(emb_user, emb_correct).item() * 100
    token_set_sim = fuzz.token_set_ratio(clean_user, clean_correct)
    levenshtein_sim = ratio(clean_user, clean_correct) * 100

    # Final score
    final_score = (semantic_sim * 0.75) + (token_set_sim * 0.15) + (levenshtein_sim * 0.10)
    final_score = max(0, min(100, round(final_score, 2)))

    # Feedback logic
    if final_score > 80:
        feedback = "Your answer is excellent! Keep up the good work. ✅"
    elif final_score > 60:
        feedback = "Your answer is good, but it can be improved by adding more details. 🔍"
    else:
        feedback = "Your answer needs improvement. Try reviewing the key points in the question. ⚡"

    # Return the results
    return jsonify({
        "final_score": final_score,
        "feedback": feedback,
        "corrected_user_answer": clean_user,
        "semantic_similarity": semantic_sim,
        "levenshtein_similarity": levenshtein_sim,
        "token_set_similarity": token_set_sim
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
