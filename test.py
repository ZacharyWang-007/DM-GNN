import os
# os.system('python main_0.75.py>>log_0.75.txt')
# os.system('python main_0.80.py>>log_0.80.txt')
# os.system('python main_0.85.py>>log_0.85.txt')
# os.system('python main_0.90.py>>log_0.90.txt')
# os.system('python main_0.95.py>>log_0.95.txt')

oslist = ['python main_0.75.py>>log_0.75.txt', 'python main_0.80.py>>log_0.80.txt', 'python main_0.85.py>>log_0.85.txt', 'python main_0.90.py>>log_0.90.txt', 'python main_0.95.py>>log_0.95.txt']
for i in range(5):
    os.system(oslist[i])

# import subprocess

# # 定义要执行的 Python 文件列表
# file_list = ['main_0.75.py', 'main_0.80.py', 'main_0.85.py', 'main_0.90.py', 'main_0.95.py']
# output_list =  ['log_0.75.py', 'log_0.80.py', 'log_0.85.py', 'log_0.90.py', 'log_0.95.py']

# # 定义输出文件名

# # 按顺序执行每个文件，并将输出重定向到文件
# for index, file in enumerate(file_list):
#     with open(output_list[index], 'a') as f:
#         subprocess.run(['python', file], stdout=f) 