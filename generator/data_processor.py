import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generator(target_size=(28, 28))
    # هذا الكود يقوم بمعالجة الصور لتتناسب مع أي Data Set خارجية
    datagen = ImageDataGenerator(
        rescale=1.255,        # تطبيع الصور
        rotation_range=10,     # تدوير عشوائي لزيادة البيانات
        zoom_range=0.1         # تكبير بسيط
    )
    print(fتم إعداد معالج الصور لأبعاد {target_size})
    return datagen

if __name__ == __main__
    create_data_generator()


   
     def load_external_data(folder_path):
    # هذا الكود يتيح للمشروع قراءة أي صور من مجلد خارجي (مثل صور المعدات)
    # وتحويلها لتنسيق يفهمه الـ Agent
    datagen = create_data_generator()
    
    external_data = datagen.flow_from_directory(
        folder_path,
        target_size=(28, 28),
        batch_size=32,
        class_mode='categorical'
    )
    print("تم ربط المجلد الخارجي بنجاح!")
    return external_data

    import tensorflow as tf

# دالة بتبني طبقة CNN بناءً على قرار الـ Agent
def build_simple_layer(filter_count):
    # بنعمل طبقة وحدة حالياً كبداية للأسبوع التاسع
    model_layer = tf.keras.layers.Conv2D(
        filters=filter_count, 
        kernel_size=(3, 3), 
        activation='relu', 
        input_shape=(28, 28, 1)
    )
    print(f"--- Layer built with {filter_count} filters successfully ---")
    return model_layer