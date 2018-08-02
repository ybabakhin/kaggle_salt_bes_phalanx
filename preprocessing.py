from keras.applications import resnet50

def preprocess_img(img, preprocessing_function):
    if preprocessing_function == 'initial':
        return img / 255.
    elif preprocessing_function == 'resnet50':
        return resnet50.preprocess_input(img)
    else:
        ValueError("Unknown Preprocessing")