import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

result = r"/blue/buyuheng/li_an.ucsb/projects/watermark-simcse/watermarking/outputs/c4/twitter-roberta-base-sentiment/128batch_30epochs/llama8-0gpt8-1-0/loss_cl0.0-tl1.0-wneg999-margin0.7/wm-c4-alpha2.0-delta0.2|0.5-sim.csv"
df = pd.read_csv(result)

df_long = pd.melt(df[['sim_ori_wm', 'sim_ori_para', 'sim_ori_spoofing', 'sim_ori_latter_spoofing']], 
# df_long = pd.melt(df[['sim_ori_pass1', 'sim_ori_pass2']], 
                  var_name='Score Type', value_name='Score')
# df_long = pd.melt(df[['human_score', 'watermarked_text_score', 'g_watermarked_text_score']], 
#                   var_name='Score Type', value_name='Score')

plt.figure(figsize=(8, 5))
sns.boxplot(x='Score Type', y='Score', data=df_long)

plt.title(f"Similarity score")
plt.xlabel('Text Type')
plt.ylabel('Score')
plt.xticks(rotation=30)

# 显示图表
plt.savefig(f"watermarking/outputs/boxplot.png", dpi=300, bbox_inches='tight')
plt.show()
print('Complete!')
