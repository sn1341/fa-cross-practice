import socket, cv2, pickle, struct
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import os
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Настройки модели
path = os.path.join(os.path.abspath(os.curdir), 'my_model.onnx')
args_confidence = 0.2
CLASSES = ['KLUBNIKA', 'raspberry']

# Загрузка модели
print("[INFO] loading model...")
net = cv2.dnn.readNetFromONNX(path)

# Функция для обновления видеопотока в интерфейсе
def update_frame():
    global data, sock, payload_size
    try:
        while len(data) < payload_size:
            packet = sock.recv(4 * 1024)
            if not packet:
                break
            data += packet

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += sock.recv(4 * 1024)

        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        frame = imutils.resize(frame, width=400)
        
        # Преобразование изображения для использования в tkinter
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        
        # Обновление изображения в Label
        video_label.config(image=image)
        video_label.image = image
        
        # Обработка кадра с использованием модели
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (32, 32)),
                                     scalefactor=1.0 / 32, size=(32, 32), 
                                     mean=(128, 128, 128), swapRB=True)
        net.setInput(blob)
        detections = net.forward()
        
        confidence = abs(detections[0][0] - detections[0][1])
        if confidence > args_confidence:
            class_mark = np.argmax(detections)
            cv2.putText(frame, CLASSES[class_mark], (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (242, 230, 220), 2)
        # Повторный вызов для обновления кадра
        root.after(10, update_frame)

    except Exception as e:
        print(f"Error: {e}")
        sock.close()

# Функция для подключения к серверу
def connect_to_server():
    global sock, data, payload_size
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 9090))
        data = b""
        payload_size = struct.calcsize("Q")
        update_frame()
    except Exception as e:
        print(f"Connection error: {e}")

# Функция для закрытия соединения
def close_connection():
    global sock
    if sock:
        sock.close()
    root.quit()

# Инициализация tkinter
root = tk.Tk()
root.title("Client Video Stream")
root.geometry("600x500")

# Элемент Label для отображения видео
video_label = Label(root)
video_label.pack()

# Кнопки для управления
connect_button = tk.Button(root, text="Connect", command=connect_to_server)
connect_button.pack()

disconnect_button = tk.Button(root, text="Disconnect", command=close_connection)
disconnect_button.pack()

# Запуск главного цикла
root.mainloop()
