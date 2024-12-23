# Image Classification for Cat and Dog using Computer Vision

## CNN (Tiny VGG)
![tiny_vgg](https://i.ibb.co/9Gp2Kf8/tiny-vgg.png)

## Random Forest
![rf](https://i.ibb.co/rxWctQs/rf.png)

## Accuracy on test set

| Model           | Accuracy | epochs | n_estimators |
|-----------------|:--------:|:------:|:------------:|
| Random Forest   |  0.6975  |        |     100      |
| Tiny VGG        |  0.7195  |   5    |              |
| Overfitted      |  0.6320  |   5    |              |
| Reduced Overfit |  0.7255  |   5    |              |
| Data Augmented  |  0.6805  |   5    |              |

- **Random Forest**: Random Forest with HOG Features (Model 0)
- **Tiny VGG**: CNN (Tiny VGG) (Model 1)
- **Overfitted**: Overfitted Model (Model 2)
- **Reduced Overfit**: Reduced Overfitting (Model 3)
- **Data Augmented**: Data Augmented Model (Model 4)
