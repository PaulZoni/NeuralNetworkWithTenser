from keras.applications.vgg16 import VGG16
from keras.models import Sequential
import ssl
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img, img_to_array
ssl._create_default_https_context = ssl._create_unverified_context


model: Sequential = VGG16()
image_path = '/Users/p.polyakov/PycharmProjects/NeuralNetworkWithTenser/src/neuralstyle/image/Green_Sea_Turtle_grazing_seagrass.jpg'
image = load_img(path=image_path, target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image, mode='tf')
yhat = model.predict(x=image)
label = decode_predictions(yhat)
print(label)
label = label[0][0]
print('%s (%.2f%%)' % (label[1], label[2]*100))

