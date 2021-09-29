# Dog Or Cat
Chỉ cần kéo thả bức bức ảnh vào, ứng dụng thú vị này sẽ phân biệt cho bạn biết con vật trong hình là chó hay mèo :D

<p align="center">
  <img src="https://media0.giphy.com/media/HwvoeGvAZfc0uAxIsL/giphy.gif?cid=790b7611e68113adc13a5ce1cf9d9b0e859225fcc03aae15&rid=giphy.gif&ct=g" />
</p>

<p align="center>">
	<i> Demo minh hoạ </i>
</p>

# How it work
Phân loại hình ảnh chó hoặc mèo là một dự án deep learning cơ bản cho những người mới bắt đầu. Tập dữ liệu dùng để xây dựng bài toán này gồm 25.000 hình ảnh với số lượng chó mèo bằng nhau.
Tập dữ liệu có sẵn trên kaggle, các bạn có thể xem và tải xuống [ở đây](https://www.kaggle.com/c/dogs-vs-cats/data)

## Đầu tiêu là huấn luyện mô hình
Các bước để huấn luyện mô hình có thể tóm tắt như sau:
1. Nạp các thư viện cần thiết
2. Xử lý dữ liệu để đưa vào huấn luyện
3. Khởi tạo các layers cho mạng neural
4. Định nghĩa hàm callbacks và một số thông số cần thiết
5. Khởi tạo tệp dữ liệu `train` và `validation`
6. Tiến hành huấn luyện

Chi tiết các bước cụ thể các bạn có thể xem ở [dogvscat.ipynb](https://github.com/KudoKhang/DogOrCat/blob/main/dogvscat.ipynb)

## Từ kết quả huấn luyện trên, xây dựng ứng dụng GUI để dự đoán hình ảnh
Đầu tiên là import các thư viện cần thiết:

```python
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import os
```

Ứng dụng có thể dự đoán được hình ảnh chó hay mèo chính là nhờ vào kết quả mình vừa huấn luyện ở bước trên:

```python
from keras.models import load_model
model = load_model('./model1_catsVSdogs.h5')
#dictionary to label all traffic signs class.
classes = {
	0:'its a cat',
	1:'its a dog'
}
```

Giao diện đồ hoạ người dùng GUI mình sẽ xây dựng nhanh nhờ thư viện `tkinter`:

```python
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('CatsVSDogs Classification')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
```

Xây dựng hàm dự đoán ảnh. (Ảnh cần được resize về 128x128 vì đó là kích thước đầu vào của ảnh khi ta huấn luyện mô hình):

```python
def classify(file_path):
	global label_packed
	image = Image.open(file_path)
	image = image.resize((128,128))
	image = numpy.expand_dims(image, axis=0)
	image = numpy.array(image)
	image = image/255
	pred = model.predict_classes([image])[0]
	sign = classes[pred]
	print(sign)
	label.configure(foreground='#011638', text=sign)
```

Hàm upload ảnh từ máy tính:

```python
def upload_image():

	try:

		file_path=filedialog.askopenfilename()

		uploaded=Image.open(file_path)

		uploaded.thumbnail(((top.winfo_width()/2.25),

		(top.winfo_height()/2.25)))

		im=ImageTk.PhotoImage(uploaded)

		sign_image.configure(image=im)

		sign_image.image=im

		label.configure(text='')

		show_classify_button(file_path)

	except:

		pass
```

Cuối cùng là hàm `show_classify_button` và cấu hình giao diện GUI để chúng ta dự đoán hình ảnh:

```python
  
def show_classify_button(file_path):

	classify_b=Button(top,text="Classify Image",

						command=lambda: classify(file_path),

						padx=10,pady=5)

	classify_b.configure(background='#364156',

						foreground='white',

						font=('arial',10,'bold'))

						classify_b.place(relx=0.79,rely=0.46)
						
upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)

upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)

sign_image.pack(side=BOTTOM,expand=True)

label.pack(side=BOTTOM,expand=True)

heading = Label(top, text="CatsVSDogs Classification",

				pady=20, font=('arial',20,'bold'))

heading.configure(background='#CDCDCD',foreground='#364156')

heading.pack()

top.mainloop()
```

# Installations
Để khởi chạy ứng dụng ta cần cài đặt những thư viện cần thiết trong `requirements.txt` bằng một dòng lệnh:
```
pip install -r requirements.txt
```

# Usage
Để chạy ứng dụng nhận biết chó hoặc mèo chúng ta chỉ cần chạy `app.py`:
```
git clone https://github.com/KudoKhang/DogOrCat
cd DogOrCat
pip install -r requirements.txt
python app.py
```

Để sử dụng GUI của Gradio:

```
python gradio-app.py
```

<p align="center">
	<img src="https://i.imgur.com/nA9xcdF.png" />
</p>

# Summary
Như đã nói ban đầu thì đây là một ứng dụng khá đơn giản, với tập dữ liệu rất lớn và đầy đủ thì độ chính xác của mô hình sau khi huấn luyện rất cao. Ở đây mình chỉ train chưa xong 2 epoch mà val_acc đã hơn 0.7 rồi. Các bạn có thể train nhiều epochs để độ chính xác được cao hơn.