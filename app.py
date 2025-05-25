from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import whisper
import shutil
import os
import logging
import uuid

app = Flask(__name__)

# Load models
# Grammar correction model
model_name = "afinnn/fluenti_grammar_correction"
tokenizer = T5Tokenizer.from_pretrained(model_name)
grammar_model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grammar_model.to(device)

# Whisper model for speech-to-text
whisper_model = whisper.load_model("base")  # can be changed to "base", "medium", etc.

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/correct', methods=['POST'])
def correct():
    """Endpoint for grammar correction"""
    data = request.get_json()
    sentence = data.get('text', '')
    
    grammar_model.eval()
    input_ids = tokenizer.encode("fix grammar: " + sentence.strip().lower(),
                               return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        output_ids = grammar_model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)
    corrected = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return jsonify({"corrected": corrected})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Endpoint for audio transcription"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    # Create temp file with original extension
    extension = os.path.splitext(audio_file.filename)[1]
    temp_file = f"temp_{uuid.uuid4().hex}{extension}"
    
    try:
        # Save uploaded file temporarily
        audio_file.save(temp_file)
        
        # Transcribe audio
        result = whisper_model.transcribe(temp_file, language="en")
        logger.info(f"Transcription successful: {result['text']}")
        
        return jsonify({"text": result['text']})
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        return jsonify({"error": str(e)}), 500
        
    finally:
        # Clean up temp file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)