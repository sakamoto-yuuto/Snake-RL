import matplotlib.pyplot as plt
from IPython import display

plt.ion() # 开启交互模式，允许动态更新图表

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf() # 清除上一帧的图
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    
    # 画具体的每一局分数
    plt.plot(scores)
    # 画平均分趋势线
    plt.plot(mean_scores)
    
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    
    # 这一句是为了让窗口刷新出来，不然会卡死
    plt.show(block=False)
    plt.pause(.1)