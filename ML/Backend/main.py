# from fastapi import FastAPI, File, UploadFile
# import tensorflow as tf
# import numpy as np
# import uvicorn
# from io import BytesIO
# import librosa

# app = FastAPI()

# # Define emotion labels (based on your dataset)
# emotion_labels = [
#     "Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"
# ]

# # Load the model
# try:
#     model = tf.keras.models.load_model("model.h5")
#     print("✅ Model loaded successfully!")
# except Exception as e:
#     print(f"❌ Model loading failed: {e}")

# def preprocess_audio(audio_bytes):
#     """Extract MFCC features, normalize, and reshape for model input"""
#     try:
#         print("🔄 Processing audio...")
#         y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)

#         # Extract 40 MFCCs
#         mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
#         mfcc_delta = librosa.feature.delta(mfcc)  # First derivative (dynamics)
#         mfcc_delta2 = librosa.feature.delta(mfcc, order=2)  # Second derivative

#         # Stack features together
#         mfccs = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

#         # Normalize features
#         mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

#         # Compute mean to get a fixed shape
#         mfccs = np.mean(mfccs, axis=1).reshape(1, -1)

#         print("✅ Audio processed successfully!")
#         return mfccs
#     except Exception as e:
#         print(f"❌ Error in audio preprocessing: {e}")
#         return None

# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     try:
#         print("🔄 Received file:", file.filename)
#         audio_bytes = await file.read()

#         processed_audio = preprocess_audio(audio_bytes)
#         if processed_audio is None:
#             return {"error": "Audio processing failed"}

#         print("🔄 Making prediction...")
#         prediction = model.predict(processed_audio)

#         # Get the predicted emotion
#         predicted_label = emotion_labels[np.argmax(prediction)]

#         print(f"✅ Prediction: {predicted_label}")

#         return {"emotion": predicted_label}
#     except Exception as e:
#         print(f"❌ Error in prediction: {e}")
#         return {"error": str(e)}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from fastapi import FastAPI, File, UploadFile
import joblib
import numpy as np
import uvicorn
from io import BytesIO
import librosa
import os

app = FastAPI()

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Pleasant Surprise", "Sad"]

# Load model
model_path = "mental_health_model.pkl"
print("🔍 Checking model file:", model_path)
print("🔍 File Exists:", os.path.exists(model_path))

try:
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
    print("🔍 Model Type:", type(model))
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

def preprocess_audio(audio_bytes):
    """Extract MFCC features and normalize"""
    try:
        print("🔄 Processing audio...")
        y, sr = librosa.load(BytesIO(audio_bytes), sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        mfccs = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        mfccs = (mfccs - np.mean(mfccs)) / np.std(mfccs)

        # Ensure correct input shape
        mfccs = np.mean(mfccs, axis=1).reshape(1, -1)

        print(f"✅ Audio processed successfully! Shape: {mfccs.shape}")
        return mfccs
    except Exception as e:
        print(f"❌ Error in audio preprocessing: {e}")
        return None

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model not loaded. Check logs for details."}

    try:
        print("🔄 Received file:", file.filename)
        audio_bytes = await file.read()
        processed_audio = preprocess_audio(audio_bytes)

        if processed_audio is None:
            return {"error": "Audio processing failed"}

        print("🔄 Making prediction...")
        prediction = model.predict(processed_audio)
        print("🔍 Raw Model Prediction:", prediction)

        if len(prediction.shape) == 1:  # If output is a single array
            predicted_index = int(prediction[0])
        else:
            predicted_index = np.argmax(prediction)

        if predicted_index >= len(emotion_labels):
            return {"error": "Invalid prediction output"}

        predicted_label = emotion_labels[predicted_index]

        print(f"✅ Prediction: {predicted_label}")
        return {"emotion": predicted_label}
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
