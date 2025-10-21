# 🚀 FastAI Course Projects

A collection of deep learning projects completed as part of the FastAI Practical Deep Learning course. Each project demonstrates different aspects of computer vision, from basic image classification to advanced techniques like transfer learning and data augmentation.

## 📚 Course Progress

- [x] Chapter 1: Getting Started with Deep Learning
- [ ] Chapter 2: Production Ready Models
- [ ] Chapter 3: Ethics and Data
- [ ] Chapter 4: Advanced Computer Vision

## 🎯 Projects

### Project 1: Sign Language MNIST Classifier

**Status:** 🚧 In Progress  
**Objective:** Build a CNN to recognize American Sign Language (ASL) letters and numbers  
**Dataset:** Sign Language MNIST (27,455 images, 24 classes)  
**Target Accuracy:** >95%  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmypocock/fastai-course-projects/blob/main/01_image_recognition/sign_language.ipynb)

**Key Techniques:**

- Convolutional Neural Networks (CNN)
- Data augmentation
- Transfer learning with MobileNetV2
- Real-time prediction capabilities

---

### Project 2: Fresh vs Rotten Fruit Classifier

**Status:** 📝 Planned  
**Objective:** Detect food waste by classifying fruit freshness  
**Dataset:** Fruits fresh and rotten (13,599 images, 6 fruit types)  
**Target Accuracy:** >92%  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jimmypocock/fastai-course-projects/blob/main/01_image_recognition/fresh_rotten_fruits.ipynb)

**Key Techniques:**

- Multi-class classification (12 classes)
- ResNet50 transfer learning
- Class imbalance handling
- Deployment considerations

---

### Future Projects

- [ ] Medical Image Analysis (Pneumonia Detection)
- [ ] Recycling Waste Classifier
- [ ] Plant Disease Detection

## 🛠️ Technologies Used

- **Deep Learning Framework:** TensorFlow 2.x / PyTorch
- **Libraries:** FastAI, NumPy, Pandas, Matplotlib
- **Development Environment:** Google Colab (free GPU)
- **Version Control:** Git & GitHub
- **Dataset Sources:** TensorFlow Datasets, Kaggle

## 📊 Results Summary

| Project | Accuracy | F1-Score | Training Time |
|---------|----------|----------|---------------|
| Sign Language MNIST | TBD | TBD | TBD |
| Fresh/Rotten Fruits | TBD | TBD | TBD |

## 🏗️ Repository Structure

```
fastai-course-projects/
├── README.md
├── 01_image_recognition/
│   ├── sign_language.ipynb
│   └── fresh_rotten_fruits.ipynb
├── 02_advanced_projects/
│   └── (future projects)
├── utils/
│   └── helper_functions.py
├── models/
│   └── (saved model files - not tracked)
└── results/
    └── (performance metrics and visualizations)
```

## 🚀 Getting Started

1. **Clone the repository:**

```bash
git clone https://github.com/jimmypocock/fastai-course-projects.git
```

2. **Open in Google Colab:**
   - Click on any "Open in Colab" badge above
   - Or go to [Google Colab](https://colab.research.google.com) and open from GitHub

3. **Run the notebooks:**
   - Each notebook is self-contained with setup instructions
   - GPU runtime recommended (Runtime → Change runtime type → GPU)

## 📈 Learning Objectives

Through these projects, I'm learning:

- ✅ Fundamentals of deep learning and neural networks
- ✅ Practical computer vision techniques
- ✅ Model optimization and hyperparameter tuning
- ✅ Handling real-world datasets
- ✅ Deployment considerations for ML models
- ✅ Ethical AI and bias detection

## 📝 Notes

- All projects are completed using free resources (Google Colab, open datasets)
- Focus on practical, socially beneficial applications
- Each notebook includes detailed explanations and comments
- Models are evaluated on unseen test sets for realistic performance metrics

## 🤝 Acknowledgments

- [FastAI Course](https://course.fast.ai/) by Jeremy Howard and Sylvain Gugger
- Dataset providers: TensorFlow Datasets, Kaggle community
- Google Colab for free GPU access

## 📫 Contact

Feel free to reach out if you have questions or suggestions!

**GitHub:** [@jimmypocock](https://github.com/jimmypocock)  
**LinkedIn:** [Jimmy Pocock](https://linkedin.com/in/jimmypocock)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** Individual datasets used in these projects may have their own licenses:

- Sign Language MNIST: CC BY-SA 4.0
- Other datasets: Check individual sources

---

*This repository is part of my journey learning deep learning through the FastAI course. Each project represents hands-on practice with real-world applications.*
