import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from generator.data_processor import create_data_generator, build_simple_layer
from agent.nas_agent import MyNASAgent

def start_my_project():
    print("--- AutoCNN-NAS: Week 10 Integration ---")
    
    # 1. تجهيز البيانات
    gen = create_data_generator()
    print("--- Step 1: Data Generator Ready (28, 28) ---")
    
    # 2. استشارة الـ Agent لاختيار أفضل تصميم
    filters_options = [16, 32, 64]
    bot = MyNASAgent(filters_options)
    chosen_filters = bot.get_best_design()
    print(f"Agent suggests using: {chosen_filters} filters for this layer")
    
    # 3. بناء الطبقة والنموذج
    print("--- Step 3: Building the actual CNN layer... ---")
    my_layer = build_simple_layer(chosen_filters)
    
    model = tf.keras.Sequential([
        my_layer,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 4. التدريب (الحل النهائي للمولد)
    print("--- Step 4: Starting Model Training on MNIST ---")
   # 4. التدريب (الحل المضمون لربط البيانات بالنموذج)
    
    try:
        # تحويل المولد إلى Iterator وسحب أول دفعة بيانات
        data_iter = iter(gen)
        X_batch, y_batch = next(data_iter)
        
        # البدء بالتدريب الفعلي
        model.fit(X_batch, y_batch, epochs=1)
        print("--- Success: Training Completed! ---")
        
    except Exception as e:
        print(f"Training failed due to: {e}")

if __name__ == "__main__":
    print(">>> Starting the NAS Engine...", flush=True)
    start_my_project() 




        