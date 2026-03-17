🩺 Automated Otitis Media Diagnosis using Otoscopic Images with Deep CNN
📌 Project Overview

This project focuses on the automatic diagnosis of Otitis Media using otoscopic images and Deep Learning techniques. A Convolutional Neural Network (CNN) model is used to analyze ear images and classify them into different categories such as normal and infected conditions.

The system helps in early detection and supports medical professionals by providing fast and accurate predictions.

🎯 Objectives

To develop an automated system for detecting Otitis Media

To improve diagnostic accuracy using deep learning

To reduce manual effort and time in medical image analysis

To assist healthcare professionals with decision support

🧠 Methodology

Collection of otoscopic image dataset

Image preprocessing (resizing, normalization, noise removal)

Feature extraction using CNN (DenseNet121)

Model training and validation

Performance evaluation based on accuracy and loss

🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

📂 Project Structure
├── dataset/
├── models/
├── training/
├── testing/
├── results/
├── app.py
├── README.md

📊 Model Details

Model: CNN (DenseNet121 for feature extraction)

Accuracy: ~96%

Loss Function: Categorical Crossentropy

Optimizer: Adam

🚀 How to Run the Project
Step 1: Clone the repository
git clone https://github.com/your-username/your-repo-name.git
Step 2: Navigate to project folder
cd your-repo-name
Step 3: Install dependencies
pip install -r requirements.txt
Step 4: Run the application
python app.py
📷 Sample Output

The model predicts whether the ear condition is:

Normal

Otitis Media

⚠️ Limitations

Requires good quality otoscopic images

Performance depends on dataset size

May not generalize to all real-world conditions

🔮 Future Scope

Integration with mobile or web applications

Real-time diagnosis using live camera

Use of larger and more diverse datasets

Improvement in model accuracy using advanced architectures

👩‍💻 Author

Your Sudheeshna Gangula

📚 References

Deep Learning for Medical Image Analysis

Research papers on Otitis Media detection using CNN

TensorFlow & Keras documentation
