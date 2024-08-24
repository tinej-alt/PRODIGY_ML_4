# Prodigy Infotech Internship Task 4 of Machine Learning
## Hand Gesture Recognition System
### Overview
The code aims to develop a hand gesture recognition model using deep learning techniques to identify and classify different hand gestures from image or video data. The model is designed to enable intuitive human-computer interaction and serve as a component for gesture-based control systems.
### Dataset
Dataset The dataset consists of near-infrared images captured by the Leap Motion sensor. It encompasses 10 unique hand gestures meticulously performed by 10 different subjects, ensuring diversity in gesture styles and orientations. Each gesture is meticulously organized into folders corresponding to the subjects (00 to 09) and specific gestures (e.g., "01_palm," "02_l," "10_down"), offering a comprehensive and structured dataset for model training and evaluation. A part of this dataset was used for training, testing and prediction.

**Dataset Link:** https://www.kaggle.com/datasets/gti-upm/leapgestrecog

### Technologies Used:
**1.TensorFlow and Keras:** TensorFlow, an open-source ML framework, and Keras, a high-level neural networks API, simplify model development.

**2.Convolutional Neural Network (CNN):** CNN architecture is employed for image-based hand gesture recognition.

**3.ImageDataGenerator:** Keras' ImageDataGenerator facilitates efficient image loading and preprocessing.

**4.Model Training:** Adam optimizer and categorical cross-entropy loss train the CNN model.

**5.Class Label Encoding:** Scikit-learn's LabelEncoder encodes class labels.

**6.Python:** Programming language for data processing and model development.

**7.NumPy:** Library for numerical computations and array operations.

**8.Matplotlib:** Library for data visualization

### Workflow Summary:

**1.Data Preparation:** Collect or use a dataset with diverse hand gesture images and Organize the dataset into subdirectories, each corresponding to a specific hand gesture class.

**2.Model Architecture:** Design a CNN architecture suitable for image classification and Choose appropriate hyperparameters, including image size, batch size, and number of epochs.

**3.Data Preprocessing:** Use ImageDataGenerator for loading and preprocessing images and Apply data augmentation to enhance model generalization.

**4.Model Training:** Train the model on the preprocessed dataset, monitoring validation performance and Adjust hyperparameters based on training results.

**6.Prediction:** Create a function to make predictions on individual images using the trained model.

**7.Save and Load:** Save the trained model for future use and Load the model when needed for making predictions on new data.

### Conclusion:
The developed hand gesture recognition model serves as a foundation for intuitive human-computer interaction and gesture-based control systems. The use of deep learning, specifically CNNs, allows the model to learn intricate patterns in hand gestures, providing a robust and accurate solution for real-world applications.
