# Real-Time-Multimodal-Emotion-Age-and-Gender-Predictor

## Overview

The **Real Time Multimodal Emotion,Age and Gender Predictor** is a machine-learning-powered web application that performs real-time analysis of facial and audio inputs. It is capable of:

1. Detecting facial emotions such as happiness, sadness, and anger from live webcam feeds.
2. Predicting a person’s age and gender from facial images.
3. Recognizing emotions from speech inputs using audio data.

The application combines multiple machine learning models to provide seamless and accurate predictions via an interactive and responsive web interface.

## Aim

The primary aim of this project is to integrate and deploy multi-modal machine learning models for real-time emotion, age, and gender prediction to demonstrate the power of artificial intelligence in understanding human behavior through visual and auditory cues.

## Project Description

The **Real-Time Multi-Model Emotion and Age Predictor** is a Flask-based web application designed to perform three core functionalities:

1. Detect facial emotions (e.g., happy, sad, angry) from live webcam feed.
2. Recognize emotions from speech inputs.
3. Predict a person's age and gender using facial images.

This project combines cutting-edge machine learning models for image and audio processing, providing real-time insights through an interactive and responsive web interface.

## Dataset Description

- **Face Expression Recognition Dataset (Kaggle)**:

  - Contains 35,887 grayscale, 48x48 pixel images of faces labeled with seven emotion categories: **Angry**, **Disgust**, **Fear**, **Happy**, **Sad**, **Surprise**, and **Neutral**.
  - Used to train the CNN model for facial emotion recognition.
  - [Dataset Link](https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset)

- **RAVDESS Dataset**:

  - The Ryerson Audio-Visual Database of Emotional Speech and Song, consisting of 7356 files.
  - Audio spectrograms were extracted using the **Librosa** library for speech emotion recognition.
  - [Dataset Link](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

- **UTKFace Dataset**:
  - A large-scale dataset containing over 20,000 images of faces, with annotations for age, gender, and ethnicity.
  - Used for training the age and gender prediction model.
  - [Dataset Link](https://www.kaggle.com/datasets/jangedoo/utkface-new)

## Methodology

### 1. Data Loading and Preprocessing

- Facial images were resized, normalized, and augmented to improve model generalization.
- Speech data was processed into spectrograms using the **Librosa** library to extract relevant audio features.

### 2. Model Training

- **Facial Emotion Recognition**:
  - A CNN-based model was trained on the Face Expression Recognition Dataset, achieving an accuracy of 64%.
- **Speech Emotion Recognition**:
  - Feature extraction using spectrograms, followed by model training, achieving an accuracy of 89%.
- **Age and Gender Prediction**:
  - Deep learning model trained on the UTKFace dataset with 76.8% accuracy.

### 3. Web App Development

- **Frontend**: Built using HTML, CSS, Bootstrap, and JavaScript.
- **Backend**: Flask framework integrated with the trained models for inference.
- **Interactive Features**:
  - Webcam feed for facial analysis.
  - Microphone input for speech emotion detection.
  - Animated background using Particleground.js.
  - Audio waveforms visualized with WaveSurfer.js.

## Accuracy and Model Performance

| **Model**                      | **Dataset Used**              | **Training Accuracy** | **Testing Accuracy** |
| ------------------------------ | ----------------------------- | --------------------- | -------------------- |
| **Facial Emotion Recognition** | Facial Expression Recognition | 64%                   | 62%                  |
| **Speech Emotion Recognition** | RAVDESS                       | 98.4%                 | 89%                  |
| **Age Classification**         | UTKFace                       | 80%                   | 76.8%                |
| **Gender Classification**      | UTKFace                       | 84%                   | 82%                  |

## Further Scope for Improvement

- Enhance the facial emotion detection model by using larger and more diverse datasets.
- Implement real-time feedback on model predictions to improve user interaction.
- Integrate more advanced architectures like transformers or pre-trained models (e.g., ResNet, VGG).
- Optimize app performance for lower latency on live predictions.

## Requirements

- **Python Version:** 3.8+
- **Libraries:**
  - Flask
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-Learn
  - TensorFlow / Keras
  - Librosa
  - OpenCV
  - Bootstrap
  - Particleground.js
  - WaveSurfer.js

## Usage

To run this project:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/PriyanshuLathi/Real-Time-Multimodal-Emotion-Age-and-Gender-Predictor.git
   ```

2. **Navigate to the project directory:**

   ```bash
   cd Real-Time-Multimodal-Emotion-Age-and-Gender-Predictor
   ```

3. **Install the required libraries**

4. **Run the Flask app:**

   ```bash
   python app.py
   ```

5. **Access the app** in your browser at `http://127.0.0.1:5000`.

## Conclusion

This project provides a comprehensive solution for real-time emotion detection, age prediction, and speech emotion analysis using deep learning techniques. It integrates multiple models to deliver accurate predictions while offering an interactive and user-friendly web application.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/PriyanshuLathi/Real-Time-Multimodal-Emotion-Age-and-Gender-Predictor/blob/main/LICENSE) file for details.

## Contact

For any questions or feedback:

- **LinkedIn**: [Priyanshu Lathi](https://www.linkedin.com/in/priyanshu-lathi)
- **GitHub**: [Priyanshu Lathi](https://github.com/PriyanshuLathi)

Contributions are welcome! Fork this repository and submit a pull request for review.

## Authors

- **Priyanshu Lathi**
