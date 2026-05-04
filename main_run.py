# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# from generator.data_processor import create_data_generator, build_simple_layer
# from agent.nas_agent import MyNASAgent

# def start_my_project():
#     print("--- AutoCNN-NAS: Week 10 Integration ---")
    
#     # 1. تجهيز البيانات
#     gen = create_data_generator()
#     print("--- Step 1: Data Generator Ready (28, 28) ---")
    
#     # 2. استشارة الـ Agent لاختيار أفضل تصميم
#     filters_options = [16, 32, 64]
#     bot = MyNASAgent(filters_options)
#     chosen_filters = bot.get_best_design()
#     print(f"Agent suggests using: {chosen_filters} filters for this layer")
    
#     # 3. بناء الطبقة والنموذج
#     print("--- Step 3: Building the actual CNN layer... ---")
#     my_layer = build_simple_layer(chosen_filters)
    
#     model = tf.keras.Sequential([
#         my_layer,
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])
    
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
#     # 4. التدريب (الحل النهائي للمولد)
#     print("--- Step 4: Starting Model Training on MNIST ---")
#    # 4. التدريب (الحل المضمون لربط البيانات بالنموذج)
    
#     try:
#         # تحويل المولد إلى Iterator وسحب أول دفعة بيانات
#         data_iter = iter(gen)
#         X_batch, y_batch = next(data_iter)
        
#         # البدء بالتدريب الفعلي
#         model.fit(X_batch, y_batch, epochs=1)
#         print("--- Success: Training Completed! ---")
        
#     except Exception as e:
#         print(f"Training failed due to: {e}")

# if __name__ == "__main__":
#     print(">>> Starting the NAS Engine...", flush=True)
#     start_my_project() 




import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os

def start_my_project():
    print("--- AutoCNN-NAS: Starting Automated Search ---")
    
    # 1. تجهيز البيانات (MNIST)
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x_train = x_train[:2000].reshape(-1, 28, 28, 1) / 255.0
    # y_train = y_train[:2000]
    # x_test = x_test[:500].reshape(-1, 28, 28, 1) / 255.0
    # y_test = y_test[:500]
# المسار للمجلد الذي حملت فيه الداتا الجديدة
    data_path = "C:/Users/PC/Desktop/MyDataset" 

    # تحميل داتا التدريب
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(28, 28),
        batch_size=32
    )

    # تحميل داتا الاختبار
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(28, 28),
        batch_size=32
    )
    best_accuracy = 0.0
    
    # سنقوم بتجربة 3 معماريات مختلفة لنرى أيهما الأفضل
    search_space = [16, 32, 64] 

    for filters in search_space:
        print(f"\n>>> Testing Architecture with {filters} filters...")
        
        # بناء النموذج بناءً على "اقتراح" البحث
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters, (3,3), activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])

        # التدريب لمرة واحدة (Epoch 1) لتقييم السرعة والدقة
        history = model.fit(x_train, y_train, epochs=1, verbose=1, validation_data=(x_test, y_test))
        
        # استخراج الدقة
        current_acc = history.history['val_accuracy'][-1]
        print(f"Result for {filters} filters: Accuracy = {current_acc*100:.2f}%")

        # حفظ الأفضل
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            model.save("best_autocnn_model.h5")
            print(f"--- NEW BEST! Saved model with {filters} filters ---")

    print("\n" + "="*40)
    print(f"FINISH: Best Accuracy achieved: {best_accuracy*100:.2f}%")
    print("The best model is saved as: best_autocnn_model.h5")
    print("="*40)

if __name__ == "__main__":
    start_my_project()