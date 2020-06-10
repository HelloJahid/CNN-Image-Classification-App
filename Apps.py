"""
@author: Md. Jahid Hasan <jahidnoyon36@gmail.com>
"""

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from functools import partial
from tkinter.filedialog import askopenfile, askopenfilename
from keras.models import load_model, Model
import cv2
import numpy as np
from keras import backend as K
import time
from PIL import Image, ImageTk

# creating main application window
root = tk.Tk()
root.geometry("720x720") # size of the top_frame
root.title("Image Classifier")



#  Frame ###########
top_frame = Frame(root, bd = 10)
top_frame.pack()

middle_frame = Frame(root, bd =10)
middle_frame.pack()

bottom_frame = Frame(root, bd = 10)
bottom_frame.pack()


notification_frame = Frame(root, bd = 10)
notification_frame.pack()

"""    User Defined Function            """

# open a h5 file from hard-disk
def open_file(initialdir='/'):

    file_path  = askopenfilename(initialdir=initialdir, filetypes = [ ('Model Weights', '*.h5' ) ]  )
    dialog_var.set("Browse Weights but not Yet Load the Weight")
    h5_var.set(file_path)

    return file_path

def load_weights():
    dialog_var.set("Loading weight.......")
    weight_path = h5_entry.get()
    global model, height, width, channel
    model = load_model(weight_path)
    model.summary()

    load_input = model.input
    input_shape= list(load_input.shape)

    height = int(input_shape[1])
    width = int(input_shape[2])
    channel = int(input_shape[3])
    print(height, width, channel)
    dialog_var.set("Weight loaded!")

    return
# open a image file from hard-disk
def open_image(initialdir='/'):
    file_path  = askopenfilename(initialdir=initialdir, filetypes = [ ('Image File', '*.*' ) ]  )
    dialog_var.set("Browse Image but not Yet Load the Image")
    img_var.set(file_path)

    image = Image.open(file_path)
    image = image.resize((320,180)) # resize image to 32x32
    photo = ImageTk.PhotoImage(image)

    img_label = Label(middle_frame, image=photo, padx=10, pady=10)
    img_label.image = photo # keep a reference!
    img_label.grid(row=3, column=1)

    return file_path

def load_image():
    dialog_var.set("Image loading.............")
    path = img_entry.get()
    global imgs

    if channel == 1:
        imgs = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        imgs = cv2.imread(path)
    imgs = cv2.resize(imgs,(height,width)) # resize image to 32x32
    imgs = imgs.reshape(1, height, width,channel).astype('float32')
    imgs = np.array(imgs) / 255
    print(imgs.shape)
    dialog_var.set("Image loaded!, Now Test it")

    return

# #####################  Test Image
def test_image():

    # train
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    K.set_value(model.optimizer.lr,1e-3) # set the learning r

    # predict
    old = time.time()
    predictions = model.predict_classes(imgs)
    new = time.time()
    print(predictions)
    print(new-old)
    result_text = "output class: "+str(predictions)
    test_result_var.set(result_text)
    dialog_var.set("Hooray, You Done It!")



"""  Top Frame  """
# tl = Label(top_frame, text="Top frame").pack()
# ##### H5 #################
btn_h5_fopen = Button(top_frame, text='Browse Weights',  command =  lambda: open_file(h5_entry.get()), bg="black", fg="white" )
btn_h5_fopen.grid(row=2, column=1)

h5_var = StringVar()
h5_var.set("/")
h5_entry = Entry(top_frame, textvariable=h5_var, width=40)
h5_entry.grid(row=2, column=2)

btn_h5_confirm = Button(top_frame, text='Load Weights',  command = load_weights , bg="black", fg="white" )
btn_h5_confirm.grid(row=2, column=4)


#######   IMAGE input
btn_img_fopen = Button(top_frame, text='Browse Image',  command =  lambda: open_image(img_entry.get()), bg="black", fg="white" )
btn_img_fopen.grid(row=7, column=1)

img_var = StringVar()
img_var.set("/")
img_entry = Entry(top_frame, textvariable=img_var, width=40)
img_entry.grid(row=7, column=2)

btn_img_confirm = Button(top_frame, text='Load Image',  command = load_image , bg="black", fg="white" )
btn_img_confirm.grid(row=7, column=4)


""" middle Frame  """
ml = Label(middle_frame, font=("Courier", 10),bg="gray", fg="white", text="Browse Image Show Below").grid(row=1, column=1)

####### Have Image show propoer here in grid



""" bottom Frame  """

# Test Image butttom
btn_test = Button(bottom_frame, text='Test Image',  command = test_image , bg="green", fg="white" )
btn_test.pack()


test_result_var = StringVar()
test_result_var.set("Your result shown here")
test_result_label = Label(bottom_frame,font=("Courier", 20), height=3, textvariable=test_result_var, bg="white", fg="purple").pack()




"""" Notification frame """
# Define Text
dialog_var = StringVar()
dialog_var.set("Welcome to AI wolrd!")


# Label frame
labelframe1 = LabelFrame(notification_frame, text="Notification Box", bg="yellow")
labelframe1.pack()

toplabel = Label(labelframe1,font=("Courier", 15), height=2, textvariable=dialog_var, fg="red", bg="lightcyan")
toplabel.pack()



# Entering the event mainloop
top_frame.mainloop()
print("finished")
