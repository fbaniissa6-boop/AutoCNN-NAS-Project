import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generator(target_size=(28, 28)):
    # هذا الكود يقوم بمعالجة الصور لتتناسب مع أي Data Set خارجية
    datagen = ImageDataGenerator(
        rescale=1.0/255, 
        rotation_range=10,
        zoom_range=0.1
    )
    print(f"--- Step 1: Data Generator Ready {target_size} ---")
    return datagen

def load_external_data(folder_path):
    # هذا الكود يتيح للمشروع قراءة أي صور من مجلد خارجي
    datagen = create_data_generator()
    external_data = datagen.flow_from_directory(
        folder_path,
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical'
    )
    print("--- External Data Linked Successfully! ---")
    return external_data

def build_simple_layer(filter_count):
    # شغل الأسبوع التاسع: بناء طبقة بناءً على قرار الـ Agent
    model_layer = tf.keras.layers.Conv2D(
        filters=filter_count, 
        kernel_size=(3, 3), 
        activation='relu', 
        input_shape=(28, 28, 1)
    )
    print(f"--- Step 3: Layer built with {filter_count} filters successfully ---")
    return model_layer 