# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import *
from tkinter import ttk
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from tkinter import filedialog
from tkinter.filedialog import *

import cv2
import keyboard as keyboard
import numpy as np
import serial
from matplotlib import pyplot as plt
import csv
import xlwt
from xlsxwriter import Workbook
from matplotlib import pyplot as plt
import os
import pandas as pd
import io
import time
from scipy.signal import find_peaks
# import pyserial
from serial.tools import list_ports

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Bye, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def selectOutputDir():
    text1.delete(1.0, END)
    outputDir = filedialog.askdirectory(parent=window)
    text1.insert(INSERT, outputDir)


def imOpen():
    text3.delete(1.0, END)
    text0.delete(1.0, END)
    fileName = askopenfilenames(parent=window)
    text0.insert(INSERT, fileName)
    fileName = format(text0.get("1.0", 'end-1c'))
    img = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img)
    plt.colorbar()
    plt.show()
    # fileName = fileName[:-3]
    # plt.savefig(fileName + 'png')
    # plt.show()
    # outputFile = format(text3.get("1.0", 'end-1c'))
    return

def plotAlong():
    global chosen_port, lastOpenedPort
    text3.insert(INSERT, "Ready")

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    window = Tk()
    window.geometry('1100x55')
    window.title("imageLinePlotter")

    # lbl0 = Label(window, text="Выбор директории выходного файла")
    # lbl0.grid(column=0, row=0)
    lbl1 = Label(window, text="X_line")
    lbl1.grid(column=0, row=0)
    lbl2 = Label(window, text="Y_line")
    lbl2.grid(column=0, row=1)
    lbl3 = Label(window, text="Coordinate")
    lbl3.grid(column=4, row=0)

    text0 = Text(width=70, height=1)
    text0.grid(column=1, row=0, sticky=W)
    text1 = Text(width=70, height=1)
    text1.grid(column=1, row=1, sticky=W)
    text2 = Text(width=6, height=1)
    text2.grid(column=5, row=0, sticky=W)
    text3 = Text(width=6, height=1)
    text3.grid(column=7, row=0, sticky=W)
    # text0.pack()

    btn0 = Button(window, text="Open Image", command=imOpen)
    btn0.grid(column=0, row=0, sticky=W)
    btn1 = Button(window, text="Output Dir ", command=selectOutputDir)
    btn1.grid(column=0, row=1, sticky=W)
    btn2 = Button(window, text="Plot along!", command=plotAlong)
    btn2.grid(column=6, row=0, sticky=W)

    # rb0 = Radiobutton(text="Y_line", value="Y_line", variable=Position_Type)
    # rb1 = Radiobutton(text="X_line", value="X_line", variable=Position_Type)

    window.mainloop()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
