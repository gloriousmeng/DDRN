# coding: utf-8
# @Time: 2024/1/26 11:38
# @Author: Hui Meng
# @FileName: utils.py
# @Software: PyCharm Community Edition
import os


def get_oldest_file(directory, threshold=10):
    # Initial variable
    oldest = None
    # Find the oldest file
    for file in os.listdir(directory):
        # Set path
        path = os.path.join(directory, file)
        # Logistic judge
        if not os.path.isdir(path) and '.py' not in file:
            modified_time = os.stat(path).st_mtime
            if oldest is None or modified_time < oldest[1]:
                oldest = (path, modified_time)
    # Delete oldest file
    if oldest is not None and len(os.listdir(directory)) > threshold:
        os.remove(oldest[0])
        print("We have delete oldest file：", oldest[0])