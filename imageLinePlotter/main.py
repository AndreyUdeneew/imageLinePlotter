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

from matplotlib import pyplot as plt
import numpy as np

fileName = ""
fileNameBG = ""
fileNameBase = ""
fileNameBaseBG = ""


def imadjust(x, a, b, c = 0, d = 255, gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Bye, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def selectOutputDir():
    text4.delete(1.0, END)
    outputDir = filedialog.askdirectory(parent=window)
    text4.insert(INSERT, outputDir)


def imOpen():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    text6.delete(1.0, END)
    text1.delete(1.0, END)
    fileName = askopenfilenames(parent=window)
    text1.insert(INSERT, fileName)
    fileName = format(text1.get("1.0", 'end-1c'))
    im = cv2.imread(fileName)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(type(im))
    plt.imshow(im)
    plt.title('im')
    # plt.colorbar()
    plt.show()
    # fileName = fileName[:-3]
    # plt.savefig(fileName + 'png')
    # plt.show()
    # outputFile = format(text3.get("1.0", 'end-1c'))
    return

def im_adjust():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    text6.delete(1.0, END)
    text1.delete(1.0, END)
    fileName = askopenfilenames(parent=window)
    text1.insert(INSERT, fileName)
    fileName = format(text1.get("1.0", 'end-1c'))
    im = cv2.imread(fileName)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if Color_channel.get() == 0:  # Red channel intended
        image = im[:, :, 0]
    elif Color_channel.get() == 1:  # Green channel intended
        image = im[:, :, 1]
    elif Color_channel.get() == 2:  # grayscale intended
        image = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    min = np.min(image)
    max = np.max(image)
    print (min)
    print(max)
    imAdjusteded = imadjust(image, np.min(image), np.max(image), 0, 255, 1)
    # imadjust(x, a, b, c = 0, d = 255, gamma=1)
    print(type(im))
    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(im)
    axes[0].set_title('original')

    axes[1].imshow(imAdjusteded)
    axes[1].set_title('adjusted')

    plt.imshow(imAdjusteded)
    plt.title('imAdjusted')
    plt.colorbar()
    plt.show()
    # fileName = fileName[:-3]
    # plt.savefig(fileName + 'png')
    # plt.show()
    # outputFile = format(text3.get("1.0", 'end-1c'))
    return


def BackgroundOpen():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    text6.delete(1.0, END)
    text2.delete(1.0, END)
    fileNameBG = askopenfilenames(parent=window)
    text2.insert(INSERT, fileNameBG)
    fileNameBG = format(text2.get("1.0", 'end-1c'))
    imBG = cv2.imread(fileNameBG)
    imBG = cv2.cvtColor(imBG, cv2.COLOR_BGR2RGB)
    plt.imshow(imBG)
    plt.title('imBG')
    # plt.colorbar()
    plt.show()
    # fileName = fileName[:-3]
    # plt.savefig(fileName + 'png')
    # plt.show()
    # outputFile = format(text3.get("1.0", 'end-1c'))


def WhiteFieldOpen():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    text6.delete(1.0, END)
    text3.delete(1.0, END)
    fileNameBase = askopenfilenames(parent=window)
    text3.insert(INSERT, fileNameBase)
    fileNameBase = format(text3.get("1.0", 'end-1c'))
    imBase = cv2.imread(fileNameBase)
    imBase = cv2.cvtColor(imBase, cv2.COLOR_BGR2RGB)
    plt.imshow(imBase)
    plt.title('imBase')
    # plt.colorbar()
    plt.show()
    # fileName = fileName[:-3]
    # plt.savefig(fileName + 'png')
    # plt.show()
    # outputFile = format(text3.get("1.0", 'end-1c'))
    return

def WhiteFieldOpen_BG():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    text6.delete(1.0, END)
    text7.delete(1.0, END)
    fileNameBaseBG = askopenfilenames(parent=window)
    text7.insert(INSERT, fileNameBaseBG)
    fileNameBaseBG = format(text7.get("1.0", 'end-1c'))
    imBaseBG = cv2.imread(fileNameBaseBG)
    imBaseBG = cv2.cvtColor(imBaseBG, cv2.COLOR_BGR2RGB)
    plt.imshow(imBaseBG)
    plt.title('imBaseBG')
    # plt.colorbar()
    plt.show()
    # fileName = fileName[:-3]
    # plt.savefig(fileName + 'png')
    # plt.show()
    # outputFile = format(text3.get("1.0", 'end-1c'))
    return

def plotMultiple():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    outputFilename = format(text4.get("1.0", 'end-1c')) + "/_output_multiple.png"
    fileNames = askopenfilenames(parent=window)
    fileNames = sorted(fileNames)
    coordinate = int(text5.get(1.0, END))

    # if Position_Type.get() == 1:  # Vertical line
    #     if Color_channel.get() == 0:  # Red channel intended
    #         line = im[:, coordinate, 2]
    #     elif Color_channel.get() == 1:  # Green channel intended
    #         line = im[:, coordinate, 1]
    #     elif Color_channel.get() == 2:  # Grayscale intended
    #         im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    #         line = im[:, coordinate]
    #
    # elif Position_Type.get() == 0:  # Horizontal line
    #     if Color_channel.get() == 0:  # Red channel intended
    #         line = im[coordinate, :, 0]
    #     elif Color_channel.get() == 1:  # Green channel intended
    #         line = im[coordinate, :, 1]
    #     elif Color_channel.get() == 2:  # Grayscale intended
    #         im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
    #         line = im[:, coordinate]

    for fileName in fileNames:
        print(fileName)
        im = cv2.imread(fileName)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if Position_Type.get() == 1:  # Vertical line
            if Color_channel.get() == 0:  # Red channel intended
                line = im[:, coordinate, 2]
            elif Color_channel.get() == 1:  # Green channel intended
                line = im[:, coordinate, 1]
        elif Position_Type.get() == 0:  # Horizontal line
            if Color_channel.get() == 0:  # Red channel intended
                line = im[coordinate, :, 0]
            elif Color_channel.get() == 1:  # Green channel intended
                line = im[coordinate, :, 1]
        plt.plot(line)
    print(outputFilename)
    plt.savefig(outputFilename)
    plt.show()
    text6.insert(INSERT, 'Ready')
    text8.insert(INSERT, 'Ready')

def plotEasy():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    outputFilename = fileName[:-3] + "_output.png"
    coordinate = int(text5.get(1.0, END))
    text6.insert(INSERT, "Ready")
    im = cv2.imread(fileName)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    print(type(im))

    if Position_Type.get() == 1:  # Vertical line
        if Color_channel.get() == 0:  # Red channel intended
            line = im[:, coordinate, 2]
        elif Color_channel.get() == 1:  # Green channel intended
            line = im[:, coordinate, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            line = im[:, coordinate]

    elif Position_Type.get() == 0:  # Horizontal line
        if Color_channel.get() == 0:  # Red channel intended
            line = im[coordinate, :, 0]
        elif Color_channel.get() == 1:  # Green channel intended
            line = im[coordinate, :, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            line = im[:, coordinate]

    plt.plot(line)
    plt.xlabel('Pixels')
    plt.ylabel('Pixels values')
    plt.title('Pixels values along the line')
    plt.savefig(outputFilename)
    plt.show()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    if Position_Type.get() == 1:  # Vertical line
        im[:, coordinate, 2] = '255'
    elif Position_Type.get() == 0:  # Horizontal line
        im[coordinate, :, 2] = '255'
    cv2.imshow(outputFilename, im)
    imageName = fileName[:-3] + "_demo.png"
    cv2.imwrite(imageName, im)
    return


def plot_normalized_adjusted_4_overlay():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    outputFilename = fileName[:-3] + "_output.png"
    coordinate = int(text5.get(1.0, END))
    print(coordinate)
    text6.insert(INSERT, "Ready")

    imNormalized = im / imBase
    # imNormalized = np.asarray(imNormalized)
    # imNormalized = imadjust(imNormalized, imNormalized.min(), imNormalized.max(), 0, 1)
    # cv2.imshow(outputFilename, imNormalized)
    # print(type(imNormalized))
    BG_normalized = imBG / imBaseBG
    # BG_normalized = imBG
    # place 4 adjustment
    imOut = imNormalized - BG_normalized

    if Position_Type.get() == 1:  # Vertical line
        if Color_channel.get() == 0:  # Red channel intended
            lineOut = imOut[:, coordinate, 0] + BG_normalized[:, coordinate, 1]
        elif Color_channel.get() == 1:  # Green channel intended
            lineOut = imOut[:, coordinate, 1] + BG_normalized[:, coordinate, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            lineOut = imOut[:, coordinate] + BG_normalized[:, coordinate]

    elif Position_Type.get() == 0:  # Horizontal line
        if Color_channel.get() == 0:  # Red channel intended
            lineOut = imOut[coordinate, :, 0] + BG_normalized[coordinate, :, 1]
        elif Color_channel.get() == 1:  # Green channel intended
            lineOut = imOut[coordinate, :, 1] + BG_normalized[coordinate, :, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            imOut = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            lineOut = imOut[coordinate, :] + BG_normalized[coordinate, :]

    lineOut = imadjust(lineOut, lineOut.min(), lineOut.max(), 0, 255)
    plt.plot(lineOut)
    plt.xlabel('Pixels')
    plt.ylabel('Pixels values')
    plt.title('Pixels values along the line')
    plt.savefig(outputFilename)
    plt.show()
    if Position_Type.get() == 1:  # Vertical line
        im[:, coordinate, 0] = '255'
    elif Position_Type.get() == 0:  # Horizontal line
        im[coordinate, :, 0] = '255'
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow(outputFilename, im)
    imageName = fileName[:-3] + "_demo.png"
    cv2.imwrite(imageName, im)
    return

def plot_normalized_adjusted():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    outputFilename = fileName[:-3] + "_output.png"
    coordinate = int(text5.get(1.0, END))
    print(coordinate)
    text6.insert(INSERT, "Ready")

    imNormalized = im / imBase
    # place 4 adjustment
    BG_normalized = imBG / imBaseBG
    # BG_normalized = imBG
    # place 4 adjustment
    imOut = imNormalized - BG_normalized

    if Position_Type.get() == 1:  # Vertical line
        if Color_channel.get() == 0:  # Red channel intended
            lineOut = imOut[:, coordinate, 0]
        elif Color_channel.get() == 1:  # Green channel intended
            lineOut = imOut[:, coordinate, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            lineOut = imOut[:, coordinate]

    elif Position_Type.get() == 0:  # Horizontal line
        if Color_channel.get() == 0:  # Red channel intended
            lineOut = imOut[coordinate, :, 0]
        elif Color_channel.get() == 1:  # Green channel intended
            lineOut = imOut[coordinate, :, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            imOut = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            lineOut = imOut[coordinate, :]

    lineOut = imadjust(lineOut, lineOut.min(), lineOut.max(), 0, 255)
    plt.plot(lineOut)
    plt.xlabel('Pixels')
    plt.ylabel('Pixels values')
    plt.title('Pixels values along the line')
    plt.savefig(outputFilename)
    plt.show()
    if Position_Type.get() == 1:  # Vertical line
        im[:, coordinate, 0] = '255'
    elif Position_Type.get() == 0:  # Horizontal line
        im[coordinate, :, 0] = '255'
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow(outputFilename, im)
    imageName = fileName[:-3] + "_demo.png"
    cv2.imwrite(imageName, im)
    return


def plot_like_Eimar():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    outputFilename = fileName[:-3] + "_output.png"
    coordinate = int(text5.get(1.0, END))
    print(coordinate)
    text6.insert(INSERT, "Ready")

    imOut = im - imBG

    if Position_Type.get() == 1:  # Vertical line
        if Color_channel.get() == 0:  # Red channel intended
            lineOut = imOut[:, coordinate, 0] + imBG[:, coordinate, 1]
        elif Color_channel.get() == 1:  # Green channel intended
            lineOut = imOut[:, coordinate, 1] + imBG[:, coordinate, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            lineOut = imOut[:, coordinate] + imBG[:, coordinate, 1]

    elif Position_Type.get() == 0:  # Horizontal line
        if Color_channel.get() == 0:  # Red channel intended
            lineOut = imOut[coordinate, :, 0] + imBG[coordinate, :, 1]
        elif Color_channel.get() == 1:  # Green channel intended
            lineOut = imOut[coordinate, :, 1] + imBG[coordinate, :, 1]
        elif Color_channel.get() == 2:  # Grayscale intended
            imOut = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
            lineOut = imOut[coordinate, :] + imBG[coordinate, :, 1]

    plt.plot(lineOut)
    plt.xlabel('Pixels')
    plt.ylabel('Pixels values')
    plt.title('Pixels values along the line')
    plt.savefig(outputFilename)
    plt.show()
    if Position_Type.get() == 1:  # Vertical line
        im[:, coordinate, 0] = '255'
    elif Position_Type.get() == 0:  # Horizontal line
        im[coordinate, :, 0] = '255'
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    cv2.imshow(outputFilename, im)
    imageName = fileName[:-3] + "_demo.png"
    cv2.imwrite(imageName, im)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    window = Tk()
    window.geometry('1350x160')
    window.title("imageLinePlotter")

    # lbl0 = Label(window, text="Выбор директории выходного файла")
    # lbl0.grid(column=0, row=0)
    lbl1 = Label(window, text="X_line")
    lbl1.grid(column=0, row=0)
    lbl2 = Label(window, text="Y_line")
    lbl2.grid(column=0, row=1)
    lbl3 = Label(window, text="Coordinate")
    lbl3.grid(column=4, row=0)

    text1 = Text(width=70, height=1)  # image
    text1.grid(column=1, row=0, sticky=W)
    text2 = Text(width=70, height=1)  # Background
    text2.grid(column=1, row=1, sticky=W)
    text3 = Text(width=70, height=1)  # Base (white field)
    text3.grid(column=1, row=2, sticky=W)
    text4 = Text(width=70, height=1)  # Output DIR
    text4.grid(column=1, row=4, sticky=W)
    text5 = Text(width=6, height=1)  # Coordinate
    text5.grid(column=5, row=0, sticky=W)
    text6 = Text(width=6, height=1)
    text6.grid(column=7, row=0, sticky=W)  # Status
    text7 = Text(width=70, height=1)  # Output DIR
    text7.grid(column=1, row=3, sticky=W)
    text8 = Text(width=70, height=1)  # Output DIR
    text8.grid(column=1, row=5, sticky=W)
    # text0.pack()

    btn1 = Button(window, text="Select Image", command=imOpen)
    btn1.grid(column=0, row=0, sticky=W)
    btn2 = Button(window, text="Select BG", command=BackgroundOpen)
    btn2.grid(column=0, row=1, sticky=W)
    btn3 = Button(window, text="Select Base", command=WhiteFieldOpen)
    btn3.grid(column=0, row=2, sticky=W)
    btn4 = Button(window, text="Output Dir ", command=selectOutputDir)
    btn4.grid(column=0, row=4, sticky=W)
    btn5 = Button(window, text="Plot easy!", command=plotEasy)
    btn5.grid(column=6, row=0, sticky=W)
    btn6 = Button(window, text="[(im/base)adjusted - (BG/baseBG)adjusted] + (BG/baseBG)adjusted",
                  command=plot_normalized_adjusted_4_overlay)
    btn6.grid(column=2, row=2, sticky=W)
    btn7 = Button(window, text="(im - BG) + (BG)green", command=plot_like_Eimar)
    btn7.grid(column=2, row=3, sticky=W)
    btn8 = Button(window, text="[(im/base)adjusted - (BG/baseBG)adjusted]adjusted",
                  command=plot_normalized_adjusted)
    btn8.grid(column=2, row=4, sticky=W)
    btn9 = Button(window, text="Select Base BG", command=WhiteFieldOpen_BG)
    btn9.grid(column=0, row=3, sticky=W)
    btn10 = Button(window, text="Select multiple", command=plotMultiple)
    btn10.grid(column=0, row=5, sticky=W)
    btn11 = Button(window, text="imadjust", command=im_adjust)
    btn11.grid(column=2, row=5, sticky=W)

    Position_Type = BooleanVar()
    rb0 = Radiobutton(text="Y_line", variable=Position_Type, value=0)
    rb0.grid(column=2, row=0, sticky=W)
    rb1 = Radiobutton(text="X_line", variable=Position_Type, value=1)
    rb1.grid(column=2, row=1, sticky=W)

    Color_channel = BooleanVar()
    rb2 = Radiobutton(text="Red Channel", variable=Color_channel, value=0)
    rb2.grid(column=4, row=1, sticky=W)
    rb3 = Radiobutton(text="Green Channel", variable=Color_channel, value=1)
    rb3.grid(column=5, row=1, sticky=W)
    rb4 = Radiobutton(text="BW image", variable=Color_channel, value=2)
    rb4.grid(column=6, row=1, sticky=W)

    window.mainloop()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
