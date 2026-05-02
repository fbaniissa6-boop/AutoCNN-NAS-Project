import tensorflow as tf
import numpy as np

# class لتمثيل الـ Agent اللي رح يختار التصميم
class MyNASAgent:
    def __init__(self, options_list):
        # تخزين الخيارات المتاحة (عدد الفلاتر مثلاً)
        self.options = options_list
        print("--- Agent is ready to think ---")

    def get_best_design(self):
        # حالياً بنختار بشكل عشوائي للتبسيط في البداية
        # بكرة بنضيف منطق الـ Reinforcement Learning هون
        choice = np.random.choice(self.options)
        
        print(f"I suggest using: {choice} filters for this layer")
        return choice

# تجربة الكود (Testing)
if __name__ == "__main__":
    # هاد الجزء بس للتأكد إن الكود شغال
    my_filters = [16, 32, 64]
    bot = MyNASAgent(my_filters)
    bot.get_best_design()