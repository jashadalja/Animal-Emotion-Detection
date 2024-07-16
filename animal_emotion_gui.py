import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# Load the trained model without the optimizer
model = load_model('pets_detection.keras', compile=False)

# Recompile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a Tkinter window
root = tk.Tk()
root.title("Animal Emotion Classifier")

# Create a frame for the GUI components
frame = tk.Frame(root)
frame.pack(pady=20)

# Function to load and preprocess the image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to upload an image
def upload_image():
    global file_path
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)

        label_img.config(image=img)
        label_img.image = img
        label_result.config(text="")

# Function to detect emotion
def detect_emotion():
    if file_path:
        preprocessed_img = preprocess_image(file_path)
        prediction = model.predict(preprocessed_img)
        predicted_class = np.argmax(prediction, axis=1)

        emotion_map = {0: "Angry", 1: "sad", 2: "Happy"}
        predicted_emotion = emotion_map[predicted_class[0]]
        label_result.config(text=f"Predicted Emotion: {predicted_emotion}")

# GUI components
label_instruction = tk.Label(frame, text="Upload an animal image to predict its emotion:")
label_instruction.pack(pady=10)

button_upload = tk.Button(frame, text="Upload Image", command=upload_image)
button_upload.pack(pady=10)

label_img = tk.Label(frame)
label_img.pack(pady=10)

button_detect = tk.Button(frame, text="Detect Emotion", command=detect_emotion)
button_detect.pack(pady=10)

label_result = tk.Label(frame, text="")
label_result.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
