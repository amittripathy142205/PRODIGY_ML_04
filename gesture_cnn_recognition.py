import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from datetime import datetime

class SimpleGestureRecognizer:
    def __init__(self):
        self.img_size = (64, 64) 
        self.num_classes = 5
        self.class_names = ['palm', 'fist', 'thumbs_up', 'peace', 'ok']
        self.model = None
        self.label_encoder = LabelEncoder()
        self.results = {'training': {}, 'predictions': [], 'accuracy': 0}
        
        self.output_dir = f"gesture_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output will be saved to: {self.output_dir}")
    
    def create_demo_data(self, samples_per_class=100):
        print("Creating demo dataset...")
        X, y = [], []
        
        for i, gesture in enumerate(self.class_names):
            for j in range(samples_per_class):
                img = np.random.rand(64, 64, 3)
                
                if gesture == 'fist':
                    img[20:44, 20:44] = 0.2  
                elif gesture == 'palm':
                    img[10:54, 10:54] = 0.8  
                elif gesture == 'thumbs_up':
                    img[10:30, 25:35] = 0.9  
                elif gesture == 'peace':
                    img[10:30, 20:25] = 0.9  
                    img[10:30, 35:40] = 0.9
                elif gesture == 'ok':
                    cv2.circle(img, (32, 32), 15, (0.9, 0.9, 0.9), 2)  
                
                X.append(img)
                y.append(i)
        
        X = np.array(X)
        y = keras.utils.to_categorical(y, self.num_classes)
        
        print(f"Created {len(X)} synthetic images")
        return X, y
    
    def create_model(self):
        model = keras.Sequential([
            layers.Conv2D(32, 3, activation='relu', input_shape=(64, 64, 3)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, activation='relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, 3, activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_and_save(self, epochs=20):
        print("Training model...")
        
        X, y = self.create_demo_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.create_model()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=32,
            verbose=1
        )
        
        model_path = os.path.join(self.output_dir, 'gesture_model.h5')
        self.model.save(model_path)
        
        encoder_path = os.path.join(self.output_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder.fit(self.class_names), f)
        
        self.results['training'] = {
            'final_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'epochs': epochs,
            'model_path': model_path
        }
        
        self.plot_and_save_history(history)
        
        self.test_and_save_predictions(X_test, y_test)
        
        print(f"Training completed! Results saved to {self.output_dir}")
        return history
    
    def plot_and_save_history(self, history):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training plot saved: {plot_path}")
    
    def test_and_save_predictions(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        
        accuracy = np.mean(predicted_classes == true_classes)
        self.results['accuracy'] = float(accuracy)
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for i in range(10):
            if i < len(X_test):
                axes[i].imshow(X_test[i])
                true_label = self.class_names[true_classes[i]]
                pred_label = self.class_names[predicted_classes[i]]
                confidence = predictions[i][predicted_classes[i]]
                
                color = 'green' if true_classes[i] == predicted_classes[i] else 'red'
                axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                                color=color, fontsize=8)
                axes[i].axis('off')
                

                self.results['predictions'].append({
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': float(confidence),
                    'correct': bool(true_classes[i] == predicted_classes[i])
                })
        
        plt.tight_layout()
        pred_plot_path = os.path.join(self.output_dir, 'predictions_sample.png')
        plt.savefig(pred_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Test Accuracy: {accuracy:.3f}")
        print(f"Predictions plot saved: {pred_plot_path}")
    
    def predict_and_save(self, image, save_image=True):
        """Predict gesture and save result"""
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image
        
        
        img_resized = cv2.resize(img, self.img_size)
        img_normalized = img_resized.astype('float32') / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        
       
        prediction = self.model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        gesture_name = self.class_names[predicted_class]
        
        result = {
            'gesture': gesture_name,
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat(),
            'all_probabilities': {self.class_names[i]: float(prediction[0][i]) for i in range(self.num_classes)}
        }
        
        if save_image:
            
            plt.figure(figsize=(8, 6))
            plt.imshow(img)
            plt.title(f'Predicted: {gesture_name}\nConfidence: {confidence:.3f}', fontsize=14)
            plt.axis('off')
            
            img_path = os.path.join(self.output_dir, f'prediction_{datetime.now().strftime("%H%M%S")}.png')
            plt.savefig(img_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            result['saved_image'] = img_path
        
        return result
    
    def webcam_recognition_with_save(self, duration_seconds=30):
        if self.model is None:
            print("No model loaded!")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        print(f"Starting webcam recognition for {duration_seconds} seconds...")
        print("Press 'q' to quit early, 's' to save current prediction")
        
        webcam_results = []
        start_time = cv2.getTickCount()
        frame_count = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = os.path.join(self.output_dir, 'webcam_recognition.avi')
        out = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            if frame_count % 10 == 0:
                result = self.predict_and_save(frame, save_image=False)
                webcam_results.append(result)
                
                text = f"{result['gesture']}: {result['confidence']:.2f}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            out.write(frame)
            
            cv2.imshow('Gesture Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or elapsed > duration_seconds:
                break
            elif key == ord('s'):
                saved_result = self.predict_and_save(frame, save_image=True)
                print(f"Saved prediction: {saved_result['gesture']} ({saved_result['confidence']:.3f})")
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
                webcam_file = os.path.join(self.output_dir, 'webcam_results.json')
        with open(webcam_file, 'w') as f:
            json.dump(webcam_results, f, indent=2)
        
        print(f"Webcam session saved: {len(webcam_results)} predictions")
        print(f"Video saved: {video_path}")
        print(f"Results saved: {webcam_file}")
        
        return webcam_results
    
    def save_final_report(self):
        """Save comprehensive results report"""
        report_path = os.path.join(self.output_dir, 'final_report.json')
        
        report = {
            'model_info': {
                'classes': self.class_names,
                'image_size': self.img_size,
                'num_classes': self.num_classes
            },
            'results': self.results,
            'timestamp': datetime.now().isoformat(),
            'output_directory': self.output_dir
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Final report saved: {report_path}")
        return report_path

def main():
    """Main execution function"""
    print("🚀 SIMPLIFIED GESTURE RECOGNITION WITH AUTO-SAVE")
    print("=" * 50)
    
    recognizer = SimpleGestureRecognizer()
    
    print("\n1. Training model...")
    history = recognizer.train_and_save(epochs=15)
    
    print("\n2. Starting webcam recognition...")
    webcam_results = recognizer.webcam_recognition_with_save(duration_seconds=20)
    
    print("\n3. Generating final report...")
    report_path = recognizer.save_final_report()
    
    print("\n COMPLETE! All results saved.")
    print(f" Check folder: {recognizer.output_dir}")
    print("\nSaved files:")
    print("- gesture_model.h5 (trained model)")
    print("- training_history.png (training plots)")
    print("- predictions_sample.png (test predictions)")
    print("- webcam_recognition.avi (recorded video)")
    print("- webcam_results.json (prediction data)")
    print("- final_report.json (complete summary)")

if __name__ == "__main__":
    main()
