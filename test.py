import gradio as gr
import numpy
from PIL import Image
from keras.models import load_model
model = load_model('./model1_catsVSdogs.h5')
classes = { 
    0:'its a cat',
    1:'its a dog',
}
def classify(img):
    image = Image.open(img)
    image = image.resize((128,128))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    image = image/255
    pred = model.predict_classes([image])[0]
    sign = classes[pred]
    print(sign)
    return sign
img = "./dogs-vs-cats/test1/100.jpg"
classify(img)



# iface = gr.Interface(classify, 
#                     gr.inputs.Image(shape=(224, 224)),
#                     gr.outputs.Textbox())

# iface.launch(share=True)