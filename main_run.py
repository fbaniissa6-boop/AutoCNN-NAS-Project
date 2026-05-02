# # استدعاء الشغل اللي عملناه في الأسابيع الماضية

# from generator.data_processor import create_data_generator
# from agent.nas_agent import MyNASAgent

# def start_my_project():
#     print("--- Starting AutoCNN-NAS Week 8 ---")
    
#     # 1. تجهيز الداتا 
#     print("Step 1: Preparing Data...")
#     gen = create_data_generator()
    
#     # 2. تشغيل الـ Agent 
#     print("Step 2: Asking the Agent for a design...")
#     filters_options = [16, 32, 64]
#     bot = MyNASAgent(filters_options)
    
#     # الـ Agent بيختار عدد الفلاتر
#     chosen_filters = bot.get_best_design()
    
#     print(f"--- Action: We will build a CNN with {chosen_filters} filters ---")

# if __name__ == "__main__":
#     start_my_project()##


# استدعاء الدوال من الملفات اللي عدلناها
from generator.data_processor import create_data_generator, build_simple_layer
from agent.nas_agent import MyNASAgent

def start_my_project():
    print("--- AutoCNN-NAS: Week 9 Integration ---")
    
    # 1. تجهيز الداتا )
    gen = create_data_generator()
    
    # 2. قرار الـ Agent 
    bot = MyNASAgent([16, 32, 64])
    chosen_filters = bot.get_best_design()
    
    # 3. بناء الطبقة فعلياً 
    print("Step 3: Building the actual CNN layer...")
    my_layer = build_simple_layer(chosen_filters)
    
    print("--- Everything is connected and working! ---")

if __name__ == "__main__":
    start_my_project()