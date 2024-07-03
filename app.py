
import json
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gdown

def load_and_prepare_image_vgg16(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize((32, 32))  # Resize to the expected size of the model
    image = np.array(image).astype('float32') / 255  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def load_and_prepare_image_my_cnn(uploaded_file):
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))
    image = np.array(image).astype('float32') / 255 
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, -1)
    return image

st.set_option('deprecation.showPyplotGlobalUse', False)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model_option = st.sidebar.selectbox(
    "Виберіть модель:",
    ("Модель на основі VGG16", "Конволюційна нейронна мережа")
)


if model_option == "Конволюційна нейронна мережа":
    # MODEL_PATH = 'my_cnn_model.h5'
    MODEL_PATH = 'https://drive.google.com/uc?id=1ypdFzGn9kndNHV0f5MfvLIg3bbN7V26Y'
    MODEL_PATH_HISTORY = 'my_cnn_model_history.json'
elif model_option == "Модель на основі VGG16":
    # MODEL_PATH = 'fashion_mnist_vgg16_model.h5'
    MODEL_PATH = 'https://drive.google.com/uc?id=1G_2WGJZOItyyQcM21QTwNgGy3dsxvpsG'
    MODEL_PATH_HISTORY = 'fashion_mnist_vgg16_model_history.json'
    

@st.cache_resource
def load_model(url):
    output = 'model.h5'
    gdown.download(url, output, quiet=False, fuzzy=True)
    model = tf.keras.models.load_model(output)
    return model
    
with open(MODEL_PATH_HISTORY, 'r') as f:
    history_data = json.load(f)

with st.sidebar:
    # Графік втрат
    fig1, ax1 = plt.subplots()
    ax1.plot(history_data['loss'], label='Тренувальні втрати')
    ax1.plot(history_data['val_loss'], label='Втрати валідації')
    ax1.set_title('Функція втрат')
    ax1.set_xlabel('Епоха')
    ax1.set_ylabel('Втрати')
    ax1.legend()
    st.pyplot(fig1)

    # Графік точності
    fig2, ax2 = plt.subplots()
    ax2.plot(history_data['accuracy'], label='Тренувальна точність')
    ax2.plot(history_data['val_accuracy'], label='Точність валідації')
    ax2.set_title('Точність')
    ax2.set_xlabel('Епоха')
    ax2.set_ylabel('Відсоток точності')
    ax2.legend()
    st.pyplot(fig2)


model = load_model(MODEL_PATH)

st.title('Класифікатор зображень одягу Fashion MNIST')

uploaded_file = st.file_uploader("Виберіть зображення...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Завантажене зображення', use_column_width=True)
    
    if  MODEL_PATH == 'https://drive.google.com/uc?id=1ypdFzGn9kndNHV0f5MfvLIg3bbN7V26Y':
        prepared_image = load_and_prepare_image_my_cnn(uploaded_file)
    else:
        prepared_image = load_and_prepare_image_vgg16(uploaded_file)
    
    predictions = model.predict(prepared_image)
    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]
    
    st.subheader(f'Передбачений клас: {predicted_class_name}')
    
    st.write("Імовірності для усіх класів:")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.2f}%")

   