import matplotlib.pyplot as plt
import numpy as np
import json
import time
from datetime import datetime, timedelta
import threading
from matplotlib.animation import FuncAnimation
import os

class TrainingVisualizer:
    def __init__(self, history_file='models/training_history.json', update_interval=10):
        self.history_file = history_file
        self.update_interval = update_interval
        self.fig = None
        self.axes = None
        self.animation = None
        self.is_running = False
        
    def create_dashboard(self):
        """创建训练仪表盘"""
        plt.style.use('dark_background')
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('🎵 HoyoMusic训练实时监控', fontsize=16, color='cyan')
        
        # 设置子图标题
        self.axes[0, 0].set_title('📉 损失函数', color='yellow')
        self.axes[0, 1].set_title('📈 准确率', color='green')
        self.axes[1, 0].set_title('⏱️ 训练进度', color='orange')
        self.axes[1, 1].set_title('📊 训练统计', color='magenta')
        
        # 设置网格
        for ax in self.axes.flat:
            ax.grid(True, alpha=0.3)
            ax.set_facecolor('black')
        
        plt.tight_layout()
        return self.fig, self.axes
    
    def load_training_data(self):
        """加载训练数据"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                return data
            else:
                return {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def update_plots(self, frame):
        """更新图表"""
        data = self.load_training_data()
        
        if not data or not data.get('loss'):
            return
        
        # 清除当前图表
        for ax in self.axes.flat:
            ax.clear()
        
        epochs = range(1, len(data['loss']) + 1)
        
        # 1. 损失函数图
        ax1 = self.axes[0, 0]
        ax1.plot(epochs, data['loss'], 'r-', label='训练损失', linewidth=2)
        if 'val_loss' in data and data['val_loss']:
            ax1.plot(epochs, data['val_loss'], 'orange', label='验证损失', linewidth=2)
        ax1.set_title('📉 损失函数', color='yellow')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('损失')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 准确率图
        ax2 = self.axes[0, 1]
        ax2.plot(epochs, data['accuracy'], 'g-', label='训练准确率', linewidth=2)
        if 'val_accuracy' in data and data['val_accuracy']:
            ax2.plot(epochs, data['val_accuracy'], 'cyan', label='验证准确率', linewidth=2)
        ax2.set_title('📈 准确率', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('准确率')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 训练进度环形图
        ax3 = self.axes[1, 0]
        current_epoch = len(data['loss'])
        
        # 假设总epoch数（可以从配置文件读取）
        try:
            with open('models/training_config.json', 'r') as f:
                config = json.load(f)
                total_epochs = config.get('epochs', 100)
        except:
            total_epochs = 100
        
        progress = min(current_epoch / total_epochs, 1.0)
        
        # 绘制进度圆环
        theta = np.linspace(0, 2 * np.pi, 100)
        r_outer = 1
        r_inner = 0.7
        
        # 背景圆环
        ax3.fill_between(theta, r_inner, r_outer, alpha=0.3, color='gray')
        
        # 进度圆环
        progress_theta = theta[:int(100 * progress)]
        if len(progress_theta) > 0:
            ax3.fill_between(progress_theta, r_inner, r_outer, color='lime', alpha=0.8)
        
        # 添加文本
        ax3.text(0, 0, f'{current_epoch}/{total_epochs}\n{progress*100:.1f}%', 
                ha='center', va='center', fontsize=12, color='white', weight='bold')
        
        ax3.set_xlim(-1.2, 1.2)
        ax3.set_ylim(-1.2, 1.2)
        ax3.set_aspect('equal')
        ax3.axis('off')
        ax3.set_title('⏱️ 训练进度', color='orange')
        
        # 4. 训练统计
        ax4 = self.axes[1, 1]
        ax4.axis('off')
        
        # 计算统计信息
        if data['loss']:
            current_loss = data['loss'][-1]
            current_acc = data['accuracy'][-1]
            best_val_loss = min(data.get('val_loss', [current_loss]))
            best_val_acc = max(data.get('val_accuracy', [current_acc]))
            
            # 训练速度（损失改善率）
            if len(data['loss']) > 10:
                recent_improvement = data['loss'][-10] - data['loss'][-1]
                improvement_rate = recent_improvement / 10
            else:
                improvement_rate = 0
            
            stats_text = f"""
📊 训练统计

🔥 当前Epoch: {current_epoch}
📉 当前损失: {current_loss:.4f}
🎯 当前准确率: {current_acc:.4f}

🏆 最佳验证损失: {best_val_loss:.4f}
🏆 最佳验证准确率: {best_val_acc:.4f}

⚡ 改善速度: {improvement_rate:.4f}/epoch
🕐 更新时间: {datetime.now().strftime('%H:%M:%S')}
            """
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    fontsize=10, color='white', va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='darkblue', alpha=0.7))
        
        ax4.set_title('📊 训练统计', color='magenta')
        
        plt.tight_layout()
    
    def start_monitoring(self):
        """开始监控训练"""
        if self.fig is None:
            self.create_dashboard()
        
        self.is_running = True
        self.animation = FuncAnimation(
            self.fig, 
            self.update_plots, 
            interval=self.update_interval * 1000,  # 转换为毫秒
            cache_frame_data=False
        )
        
        plt.show()
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        if self.animation:
            self.animation.event_source.stop()
    
    def save_final_report(self, output_path='training_report.png'):
        """保存最终训练报告"""
        self.update_plots(0)  # 更新一次图表
        self.fig.savefig(output_path, dpi=300, bbox_inches='tight', 
                        facecolor='black', edgecolor='none')
        print(f"📊 训练报告已保存到: {output_path}")

def start_training_monitor():
    """启动训练监控器（独立进程）"""
    visualizer = TrainingVisualizer()
    visualizer.start_monitoring()

if __name__ == "__main__":
    start_training_monitor()