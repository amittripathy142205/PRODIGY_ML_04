# PRODIGY_ML_04
✋ Hand Gesture Recognition Using Machine Learning
This project focuses on building a Hand Gesture Recognition model that can accurately detect and classify hand gestures from image or video data. The system is designed to enhance human-computer interaction through gesture-based control systems, such as virtual interfaces, touchless control, and assistive technologies.

📌 Objective
To develop a robust machine learning model that can recognize hand gestures using image data from the LeapGestRecog Dataset, enabling intuitive and contactless control mechanisms.

📊 Dataset
Source: LeapGestRecog Dataset (Kaggle)
Description: The dataset contains over 20,000 grayscale images of 10 different hand gestures captured using a Leap Motion sensor.
Classes: 10 unique gestures (e.g., swipe left, swipe right, etc.)
Structure: Organized by subject, gesture type, and hand movement.

🛠️ Project Workflow
Data Preprocessing:
Image resizing and normalization
Label encoding and dataset splitting
Model Development:
Convolutional Neural Network (CNN) built using TensorFlow/Keras
Trained on gesture images with data augmentation
Evaluation:
Accuracy, Confusion Matrix, and Loss/Accuracy plots
Futur Integration:
Real-time gesture detection using webcam input (OpenCV)
Deployment in gesture-controlled applications

🎯 Results
Achieved high classification accuracy in distinguishing between various hand gestures, proving the model’s potential for real-world gesture recognition tasks.

🚀 Future Improvements
Integrate real-time video stream input
Expand dataset with new gestures or lighting conditions
Deploy model with a GUI using Streamlit or Flask
