import tensorflow as tf

def prepare_mnist_data():
    # تحميل البيانات (MNIST) التي سنستخدمها لضبط الوقت والأوزان
    print("جاري تحميل بيانات MNIST من السحاب...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # تحويل البيانات (Normalization) لسهولة معالجة الأوزان
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    print(f"تم التجهيز بنجاح! حجم بيانات التدريب: {x_train.shape}")
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    prepare_mnist_data() 
    