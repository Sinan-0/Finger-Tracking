# Classifier using handwritten

We want to train a classifier that recognize a drawn digit from 0 to 9. In this folder, we create the data by drawing digits ourselves. Indeed, we use `draw_digit.py` to manually draw digits. We draw 5 images per digits and then perform data augmentation to have more data to train on.

# Decription of the files

### [data](data/)

Drawn digits. 5 images per digit

### [data_augmented](data_augmented/)

Resulting images from data augmentation : we rotated slighlty the drawn digits to have more data and be more robust.

### [classifier.ipynb](classifier.ipynb)

Notebook where we perform data augmentation and train the classifier

### [draw_digit.py](draw_digit.py)

Scipt used to create the data. It has 3 arguments :
- `digit` : the digit you want to draw
- `number` : the number of the image, start with 0 
- `background_color` : the color of the background, 'black' or 'white'

Example, if you want to draw the first image of an `2` and you have a black background, run :

```
python draw_digit.py --digit 2 --number 0 --background_color black
```

Now you drew the first digit of a 2, now if you want to draw a second one run :

```
python draw_digit.py --digit 2 --number 1 --background_color black
```

### [finger_template2.jpg](finger_template2.jpg)

Template image used to track the finger

### [model_handwritten](model_handritten)

Saved trained model
