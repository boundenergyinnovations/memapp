from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import speech_recognition as sr
import os
from openai import OpenAI
import numpy as np
from scipy.spatial.distance import cosine
import json
import tempfile
from werkzeug.utils import secure_filename
import logging
from pydub import AudioSegment
import io
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class VectorDatabase:
    def __init__(self, file_path='vector_db.json'):
        self.file_path = file_path
        self.data = self.load_database()

    def load_database(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                return json.load(f)
        return {'texts': [], 'embeddings': []}

    def save_database(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f)

    def add_entry(self, text, embedding):
        self.data['texts'].append(text)
        self.data['embeddings'].append(embedding)
        self.save_database()

    def find_similar(self, query_embedding, top_k=3):
        if not self.data['embeddings']:
            return []
        
        similarities = [
            1 - cosine(query_embedding, np.array(emb))
            for emb in self.data['embeddings']
        ]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.data['texts'][i] for i in top_indices]

@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'front-speech.html')

# Initialize vector database
vector_db = VectorDatabase()

def get_embedding(text):
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def convert_webm_to_wav(webm_path):
    """Convert WebM audio to WAV format using pydub."""
    try:
        # Load the audio file
        audio = AudioSegment.from_file(webm_path, format="webm")
        
        # Export as WAV
        wav_path = webm_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format="wav")
        
        return wav_path
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        raise

@app.route('/save-recording', methods=['POST'])
def save_recording():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Create temporary directory if it doesn't exist
        temp_dir = tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        # Save the original audio file
        orig_path = os.path.join(temp_dir, secure_filename(f"recording_{os.urandom(8).hex()}.webm"))
        audio_file.save(orig_path)

        try:
            # Convert to WAV
            wav_path = convert_webm_to_wav(orig_path)

            # Convert speech to text
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)

            # Get embedding and store in database
            embedding = get_embedding(text)
            vector_db.add_entry(text, embedding)

            return jsonify({
                'success': True,
                'message': 'Recording processed successfully',
                'text': text
            })

        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError as e:
            logger.error(f"Speech recognition error: {str(e)}")
            return jsonify({'error': 'Error with speech recognition service'}), 500
        finally:
            # Clean up temporary files
            if os.path.exists(orig_path):
                os.remove(orig_path)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Error processing recording: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/query', methods=['POST'])
def query_database():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save and convert the audio file
        temp_dir = tempfile.gettempdir()
        orig_path = os.path.join(temp_dir, secure_filename(f"query_{os.urandom(8).hex()}.webm"))
        audio_file.save(orig_path)

        try:
            # Convert to WAV
            wav_path = convert_webm_to_wav(orig_path)

            # Convert speech to text
            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
                query_text = recognizer.recognize_google(audio)

            # Get embedding for the query
            query_embedding = get_embedding(query_text)
            
            # Find similar texts
            similar_texts = vector_db.find_similar(query_embedding)
            
            # Create context from similar texts
            context = "\n".join(similar_texts)
            
            # Query OpenAI with context
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"Context from database:\n{context}\n\nAnswer the user's question based on this context."
                    },
                    {
                        "role": "user",
                        "content": query_text
                    }
                ],
                model="gpt-4o-mini",
                max_tokens=150
            )
            
            answer = response.choices[0].message.content

            return jsonify({
                'success': True,
                'answer': answer,
                'similar_texts': similar_texts,
                'query_text': query_text
            })

        finally:
            # Clean up temporary files
            if os.path.exists(orig_path):
                os.remove(orig_path)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        # Generate speech using OpenAI's text-to-speech API
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=data['text']
        )

        # Create a temporary file to store the audio
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        temp_file.close()

        # Save the audio to the temporary file
        response.stream_to_file(temp_file.name)

        # Send the file
        return send_file(
            temp_file.name,
            mimetype='audio/mpeg',
            as_attachment=True,
            download_name='response.mp3'
        )

    except Exception as e:
        logger.error(f"Error generating speech: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
