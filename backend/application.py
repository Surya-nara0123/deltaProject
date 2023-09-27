import os
from flask import Flask, request, jsonify
from googletrans import Translator, LANGUAGES

app = Flask(__name__)

translator = Translator()
import requests
import json

url = 'http://localhost:5000/translate'  # Replace with the correct URL

data = {
    "sentence": "Hello, World!",
    "language": "fr"  # Replace with the target language
}

headers = {'Content-Type': 'application/json'}


@app.route('/translate', methods=['POST'])
def translate_text():
    if request.method == "POST":
        data = request.get_json()

        sentence = data.get("sentence")

        # Detect the language of the input text
        detected_language = translator.detect(sentence).lang

        # Translate the input text to the target language
        target_language = data.get("language")
        translated_text = translator.translate(sentence, src=detected_language, dest=target_language).text
        print(translated_text)
        response_data = {
            "input_sentence": sentence,
            "detected_language": LANGUAGES[detected_language],
            "translated_sentence": translated_text,
        }

        return jsonify(response_data), 201

if __name__ == '__main__':
    app.run(debug=True)
