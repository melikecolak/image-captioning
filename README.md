# Image Captioning using CNN-LSTM on Flickr8k

Welcome to my Image Captioning project repository! This project aims to generate descriptive captions for images using a deep learning model that integrates Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. My model is trained on the Flickr8k dataset, leveraging the power of visual and sequential data to produce accurate and meaningful image descriptions.
![Image Captioning](https://github.com/melikecolak/image-captioning/assets/73293751/12d0f6c3-dda4-4e8d-a353-195306033fce)

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Acknowledgements](#acknowledgements)

## Introduction

Image captioning is a complex task that involves generating textual descriptions for images. This project utilizes a hybrid CNN-LSTM model to achieve this, providing a comprehensive solution that can understand and describe images in natural language.

## Model Architecture

Our model consists of two main components:

1. **CNN (Convolutional Neural Network)**: Extracts features from images using a pre-trained VGG16 model.
2. **LSTM (Long Short-Term Memory)**: Generates captions based on the extracted image features.

![Model Architecture](https://github.com/melikecolak/image-captioning/assets/73293751/c6126250-2715-4382-a420-e0922ee8fb53)

## Dataset

The project uses the Flickr8k dataset, which contains 8,000 images and five captions per image. This dataset is widely used for image captioning tasks and provides a solid foundation for training and evaluating our model.

### Citation

```bibtex
@inproceedings{flickr8k,
  title={Framing image description as a ranking task: Data, models and evaluation metrics},
  author={Hodosh, Micah and Young, Peter and Hockenmaier, Julia},
  booktitle={Journal of Artificial Intelligence Research},
  volume={47},
  pages={853--899},
  year={2013}
}
```

## Training

The model is trained using the following parameters:

- **Epochs**: 100
- **Batch Size**: 32
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

After 100 epochs, the model achieved a training loss of 1.446, indicating effective learning and generalization. You can reach my model from this [Google Drive link.](https://drive.google.com/drive/folders/1FSA1l002UIyRIqFqR18G8Opmpd9I21B_?usp=sharing)


## Evaluation

The model's performance is evaluated using BLEU scores:

- **BLEU-1**: 0.516880
- **BLEU-2**: 0.293009

These scores reflect the model's ability to generate accurate and contextually relevant captions.

## Results

Here is an example of the actual vs. predicted captions:
![image](https://github.com/melikecolak/image-captioning/assets/73293751/6618d4bf-f390-458b-98f9-a302a23086ea)

## Acknowledgements

I would like to thank the creators of the Flickr8k dataset and the authors of the various papers that inspired this project:

- Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing image description as a ranking task: Data, models and evaluation metrics. *Journal of Artificial Intelligence Research*, 47, 853-899.
- Mao, J., et al. (2016). Flickr30k Entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. *Proceedings of the IEEE conference on computer vision and pattern recognition*, 2641-2649.

---

Feel free to explore, contribute, and enjoy generating captions for your images! If you have any questions or feedback, please don't hesitate to reach out.
