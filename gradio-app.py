import gradio as gr
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import load_model

model = load_model('./model1_catsVSdogs.h5')
classes = ["IT'S CAT", "IT'S DOG"]

def classify(img):
    img = cv2.resize(img, (128,128))
    img = img.astype("float") / 255.0
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    label = classes[preds.argmax()]
    # print(label)
    return label
    # return {classes[i]: float(preds[i]) for i in range(1)}

# path = "./dogs-vs-cats/test1/2.jpg"
# img = cv2.imread(path)
# classify(img)


iface = gr.Interface(
    classify, 
    gr.inputs.Image(shape=(224, 224)),
    gr.outputs.Label(num_top_classes=2),
    capture_session=True,
    examples=[
        ["./dogs-vs-cats/test1/1.jpg"],
        ["./dogs-vs-cats/test1/2.jpg"],
        ["./dogs-vs-cats/test1/3.jpg"],
        ["./dogs-vs-cats/test1/4.jpg"],
        ["./dogs-vs-cats/test1/5.jpg"],
        ["./dogs-vs-cats/test1/6.jpg"],
        ["./dogs-vs-cats/test1/7.jpg"],
    ])

if __name__ == "__main__":
    iface.launch()