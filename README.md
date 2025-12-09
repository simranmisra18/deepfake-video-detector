# Adversarial Machine Learning for Robust Deepfake Video Detection

This project focuses on creating a deepfake detection system
that can accurately identify manipulated videos and assess its
resilience against various adversarial attacks. The system integrates computer vision, deep learning, and adversarial defense
methods to maintain reliability against subtle perturbations
aimed at deceiving models. Two architectures were developed
and evaluated: Xception + BiGRU, a hybrid deep model that
captures both spatial and temporal patterns, and Meso4 +
BiGRU, a lightweight CNN + RNN hybrid model designed
for computational efficiency.
The Xception + BiGRU model employs the Xception
network, pre-trained on ImageNet, as a frame-level feature
extractor to capture detailed spatial information. The extracted features are subsequently processed by a Bidirectional
Gated Recurrent Unit (BiGRU), which examines the temporal
evolution of facial features across frames. Its bidirectional
architecture allows the model to learn dependencies in both
forward and backward directions, improving its ability to
detect subtle inconsistencies across consecutive frames that
are characteristic of deepfake videos.
On the other hand, the Meso4 + BiGRU model was implemented as a faster and simpler baseline. It uses four convolutional layers with progressively increasing filters, followed
by max pooling, dropout, and dense layers for classification.
Although it does not explicitly model temporal dependencies,
it effectively captures mesoscopic features that differentiate
genuine faces from manipulated ones at the spatial level. Both
models were tested on clean and adversarially perturbed data
to analyze robustness. Adversarial attacks, including the Fast
Gradient Sign Method (FGSM), Projected Gradient Descent
(PGD), and 2D-Malafide, were employed, followed by adversarial retraining to enhance the models’ resilience against
such perturbations. 

The following techniques and tools are employed:
• CNNs (MesoNet, Xception) – For spatial feature extraction.
• RNNs (GRU/BiGRU) – For temporal feature learning
across frames.
• Noise Augmentation – Gaussian, blur, compression to test
robustness.
• Adversarial Augmentation – 2D-Malafide learned filter
to generate perturbed fake frames for robustness experiments.
• Transfer Learning – Xception pretrained weights for
feature reuse.
• Web Deployment (Streamlit) – Upload interface, frame
previews, real-time predictions.

<img width="267" height="368" alt="image" src="https://github.com/user-attachments/assets/22ce3e46-9e3d-47e1-8b30-c75ef731e475" />
