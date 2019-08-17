from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.layers import Input
from os import listdir
from os import path
from pickle import dump

image = load_img('/Users/p.polyakov/Documents/data_set/Flickr8k/Flicker8k_Dataset/667626_18933d713e.jpg')


def extract_features(directory):
    in_layer = Input(shape=(224, 224, 3))
    model = VGG16(include_top=False, input_tensor=in_layer)
    model.summary()
    features = dict()
    for name in listdir(directory):
        filename = path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
        print('>%s' % name)
    return features


directory = '/Users/p.polyakov/Documents/data_set/Flickr8k/Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
dump(features, open('/Users/p.polyakov/Documents/data_set/features.pkl', 'wb'))
