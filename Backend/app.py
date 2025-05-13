
###############################################################################################################################

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch
from flask_cors import CORS
import base64
import os
import traceback
import requests
import re
from flask_mysqldb import MySQL
from config import Config
from werkzeug.utils import secure_filename
import mysql.connector

app = Flask(__name__)
# app.config.from_object(Config)
# CORS(app)



db = mysql.connector.connect(
    host="sql305.infinityfree.com",
    user="if0_38966515",       # Replace with your DB user
    password="Haseebay12345",   # Replace with your DB password
    database="if0_38966515_child_profiles_db",
)
@app.route('/test-db')
def test_db():
    cursor = db.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    return {"tables": tables}

# MySQL Initialization
# mysql = MySQL(app)
# with app.app_context():
#     try:
#         cur = mysql.connection.cursor()
#         cur.execute("SELECT 1")
#         print("✅ Successfully connected to the MySQL database!")
#         cur.close()
#     except Exception as e:
#         print("❌ Failed to connect to the MySQL database:", e)

# # Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("tokenizer")
# model = AutoModelForQuestionAnswering.from_pretrained("model")
# embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# # Load Q&A dataset
# df = pd.read_excel("autism_faqs.xlsx")
# questions = df["Question"].fillna("").tolist()
# answers = df["Answer"].fillna("").tolist()
# question_embeddings = embedding_model.encode(questions, convert_to_tensor=True)

# # Define CARS categories
# CARS_CATEGORIES = [
#     "Relationship with others", "Imitation skills", "Emotional responses", "Body usage",
#     "Object usage", "Adaptation to change", "Visual response", "Auditory response",
#     "Taste, smell, and tactile response", "Anxiety and fear", "Verbal communication",
#     "Non-verbal communication", "Activity level", "Intellectual response", "General impressions"
# ]

# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_question = data.get("message", "").lower().strip()

#     thank_keywords = ["thank", "thanks", "thank you", "shukriya", "thnx", "appreciate"]
#     if any(keyword in user_question for keyword in thank_keywords):
#         return jsonify({"reply": "You're welcome! Let me know if you have more questions related to Autism.😊"})

#     how_are_you_keywords = ["how are you", "how r u", "how's it going", "how are u"]
#     if any(keyword in user_question.lower() for keyword in how_are_you_keywords):
#         return jsonify({"reply": "I'm just an Educare bot, but I'm here to help you! 😊 How can I assist you today?"})

#     input_embedding = embedding_model.encode(user_question, convert_to_tensor=True)
#     scores = torch.nn.functional.cosine_similarity(input_embedding, question_embeddings)
#     best_score = torch.max(scores).item()
#     best_index = torch.argmax(scores).item()

#     if best_score < 0.5:
#         return jsonify({"reply": "Sorry, I cannot answer that question. You can ask me any question about Autism."})

#     return jsonify({"reply": answers[best_index]})

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

# def is_valid_name(name):
#     return bool(re.match(r'^[A-Za-z ]+$', name))

# def is_valid_password(password):
#     return len(password) >= 6 and any(char.isdigit() for char in password)

# @app.route('/api/add_child', methods=['POST'])
# def add_child():
#     try:
#         data = request.get_json()
#         name = data.get('name')
#         password = data.get('password')
#         age = data.get('age')
#         image_base64 = data.get('image_base64')

#         if not name or not password or age is None:
#             return jsonify({'success': False, 'message': 'Name, password, and age are required.'}), 400

#         if not is_valid_name(name):
#             return jsonify({'success': False, 'message': 'Name must only contain alphabets and spaces.'}), 400

#         if not isinstance(age, int) or age < 0 or age > 13:
#             return jsonify({'success': False, 'message': 'Age must be between 0 and 13.'}), 400

#         if not is_valid_password(password):
#             return jsonify({'success': False, 'message': 'Password must be at least 6 characters and contain a digit.'}), 400

#         cur = mysql.connection.cursor()
#         cur.execute("SELECT * FROM child_profiles WHERE name = %s AND password = %s", (name, password))
#         if cur.fetchone():
#             return jsonify({'success': False, 'message': 'Profile with this name and password already exists.'}), 400

#         image_data = base64.b64decode(image_base64) if image_base64 else None
#         cur.execute("""
#             INSERT INTO child_profiles (name, password, age, image_blob)
#             VALUES (%s, %s, %s, %s)
#         """, (name, password, age, image_data))
#         mysql.connection.commit()
#         cur.close()

#         return jsonify({'success': True, 'message': 'Child profile saved successfully.'}), 200

#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({'success': False, 'message': 'Server error: Unable to save profile.'}), 500

# @app.route('/assess_autism', methods=['POST'])
# def assess_autism():
#     data = request.json
#     scores = data.get("scores", [])

#     if len(scores) != 15:
#         return jsonify({"error": "Invalid input, 15 scores required"}), 400

#     total_score = sum(scores)
#     severity = "Severe Autism" if total_score >= 36 else "Moderate Autism" if 30 <= total_score < 36 else "No Autism or Mild Developmental Delay"
#     deficient_areas = [CARS_CATEGORIES[i] for i, score in enumerate(scores) if score >= 3]

#     return jsonify({
#         "total_score": total_score,
#         "severity": severity,
#         # "deficient_areas": deficient_areas
#     })

# # Replace this with your Hugging Face API token
# HUGGING_FACE_API_TOKEN = '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'
# # MODEL_NAME = 'tiiuae/falcon-7b-instruct'  # Or choose another small model

# # API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
# # HEADERS = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}


# # @app.route('/generate', methods=['POST'])
# # def generate_content():
# #     data = request.json
# #     screen_title = data.get('screen_title')

# #     if not screen_title:
# #         return jsonify({'error': 'screen_title is required'}), 400

# #     prompt = f"Educate an autistic child about {screen_title} in simple, engaging, and friendly language with examples."

# #     payload = {
# #         "inputs": prompt,
# #         "options": {"use_cache": True},
# #         "parameters": {"max_new_tokens": 200}
# #     }

# #     response = requests.post(API_URL, headers=HEADERS, json=payload)

# #     if response.status_code != 200:
# #         return jsonify({'error': 'Failed to generate content'}), 500

# #     try:
# #         generated_text = response.json()[0]['generated_text']
# #     except Exception as e:
# #         return jsonify({'error': 'Parsing Hugging Face response failed'}), 500

# #     return jsonify({
# #         "title": screen_title,
# #         "content": generated_text
# #     })


# # HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/gpt2"  # You can switch to a better model
# # HUGGING_FACE_API_TOKEN = "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"  # Replace with your token

# headers = {"Authorization": f"Bearer {HUGGING_FACE_API_TOKEN}"}

# # @app.route('/generate', methods=['POST'])
# # def generateContent():
# #     data = request.json
# #     lesson_title = data.get("lesson_title")
# #     screen_number = data.get("screen_number")

# #     prompt = f"Create child-friendly autism lesson screen content for the topic '{lesson_title}' (screen {screen_number + 1}). Include simple language and examples."

# #     response = requests.post(
# #         HUGGING_FACE_API_URL,
# #         headers=headers,
# #         json={"inputs": prompt}
# #     )

# #     result = response.json()

# #     if isinstance(result, dict) and result.get("error"):
# #         return jsonify({"error": result["error"]}), 500

# #     generated_text = result[0]["generated_text"]
# #     return jsonify({"content": generated_text})


# HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# @app.route('/generate', methods=['POST'])
# def generateContent():
#     data = request.json
#     lesson_title = data.get("lesson_title")

#     prompt = f"""
#     Create child-friendly autism lesson content about '{lesson_title}'. Use:
#     - Short sentences
#     - Concrete examples
#     - Positive reinforcement language
#     """

#     try:
#         response = requests.post(
#             HUGGING_FACE_API_URL,
#             headers=headers,
#             json={
#                 "inputs": prompt,
#                 "parameters": {
#                     "max_new_tokens": 300,
#                     "temperature": 0.7,  # Balances creativity vs focus
#                     "do_sample": True
#                 }
#             },
#             timeout=30  # Prevent hanging
#         )

#         if response.status_code != 200:
#             return jsonify({"error": "Model unavailable"}), 503

#         generated_text = response.json()[0]["generated_text"]
        
#         # Clean up output
#         cleaned_text = generated_text.split("Examples:")[0]  # Get first part
#         return jsonify({
#             "content": cleaned_text,
#             "visual_cue": suggest_visual_cue(lesson_title)  # Add visuals
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# def suggest_visual_cue(topic):
#     """Map topics to emojis/icons"""
#     cues = {
#         "anger": "🌋",
#         "sharing": "🧸➡️👧",
#         "turn-taking": "🔄",
#         "voice volume": "🔊→🔉"
#     }
#     return cues.get(topic.lower(), "❓")







# #  generator = pipeline("text-generation", model="gpt2")  # You can replace with another model if needed
# # generator = pipeline("text2text-generation", model="google/flan-t5-base")

# # @app.route("/get_lesson_screen/<string:title>", methods=["GET"])
# # def get_lesson_content(title):
# #     prompt = f"You are teaching an autistic child. Please explain this topic in simple terms: {title}"
# #     try:
# #         # result = generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
# #         result = generator(prompt, max_length=200)[0]['generated_text']

# #         return jsonify({"content": result})
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
