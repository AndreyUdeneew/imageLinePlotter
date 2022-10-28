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
    outputDir = filedialog.askdirectory(parent=window)
    outputFile = outputDir + '/outputCSV.csv'
    print(outputFile)
    text1.insert(INSERT, outputDir)
    print(text2.get(Text))


def imOpen():
    text1.insert(INSERT, "fileDir is:")
    text3.delete(1.0, END)
    global chosen_port, lastOpenedPort

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
