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
from matplotlib.pyplot import subplots_adjust

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
    # imageName = fileName[:-3] + "_demo.png"
    # cv2.imwrite(imageName, im)
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

def R2G():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    print("R/G")
    deltaRed = cv2.subtract(im[:, :, 0], imBG[:, :, 0])
    deltaGreen = cv2.subtract(im[:, :, 1], imBG[:, :, 1])
    Ratio = cv2.divide(deltaRed, deltaGreen)
    min = np.min(Ratio)
    max = np.max(Ratio)
    print(min)
    print(max)
    Ratio_adjusted = imadjust(Ratio, np.min(Ratio), np.max(Ratio), 0, 255, 1)
    plt.imshow(Ratio_adjusted, cmap='gray', vmin=0, vmax=255)
    plt.show()
    # cv2.imshow("ratio.bmp", deltaRed)
    # cv2.imshow("ratio.bmp", deltaGreen)
    # cv2.imshow("difference.bmp", Ratio)
    # cv2.imshow("ratio.bmp", Ratio_adjusted)

    return


def R_G():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    print("R-G")
    deltaRed = cv2.subtract(im[:, :, 0], imBG[:, :, 0])
    deltaGreen = cv2.subtract(im[:, :, 1], imBG[:, :, 1])
    DIFFERENCE = cv2.subtract(deltaRed, deltaGreen)
    # DIFFERENCE = cv2.subtract(im, imBG)[:, :, 0]
    min = np.min(DIFFERENCE)
    max = np.max(DIFFERENCE)
    print(min)
    print(max)
    DIFFERENCE_adjusted = imadjust(DIFFERENCE, np.min(DIFFERENCE), np.max(DIFFERENCE), 0, 255, 1)

    outputFilename = fileName[:-3] + "_output.png"
    coordinate = int(text5.get(1.0, END))

    if Position_Type.get() == 1:  # Vertical line
        if Color_channel.get() == 0:  # Red channel intended
            line_deltaRed = deltaRed[:, coordinate]
            line_deltaGreen = deltaGreen[:, coordinate]
            line_DIFFERENCE_adjusted = DIFFERENCE_adjusted[:, coordinate]
            # line_Ratio_adjusted = Ratio_adjusted[:, coordinate]

    elif Position_Type.get() == 0:  # Horizontal line
            line_deltaRed = deltaRed[coordinate, :]
            line_deltaGreen = deltaGreen[coordinate, :]
            line_DIFFERENCE_adjusted = DIFFERENCE_adjusted[coordinate, :]
            # line_Ratio_adjusted = Ratio_adjusted[coordinate, :]

    fig = plt.figure()
    ax_1 = fig.add_subplot(2, 2, 1)
    ax_2 = fig.add_subplot(2, 2, 2)
    ax_3 = fig.add_subplot(2, 2, 3)
    ax_4 = fig.add_subplot(2, 2, 4)

    ax_1.set(title='deltaRED', xticks=[], yticks=[])
    ax_2.set(title='difference_adjusted', xticks=[], yticks=[])
    ax_3.set(title='plot_IM', xticks=[], yticks=[], xlabel='Pixels', ylabel='Pixels Values')
    ax_4.set(title='plot_difference_adjusted', xticks=[], yticks=[], xlabel='Pixels', ylabel='Pixels Values')

    ax_1.imshow(deltaRed, cmap='gray')
    ax_2.imshow(DIFFERENCE_adjusted, cmap='gray', vmin=0, vmax=255)
    ax_3.plot(line_deltaRed)
    ax_4.plot(line_DIFFERENCE_adjusted)

    plt.savefig(outputFilename)
    plt.show()
    # cv2.imshow("ratio.bmp", deltaRed)
    # cv2.imshow("ratio.bmp", deltaGreen)
    # cv2.imshow("difference.bmp", DIFFERENCE)
    # cv2.imshow("difference.bmp", DIFFERENCE_adjusted)
    return

def comparison():
    global fileName, fileNameBase, fileNameBaseBG, fileNameBG, imBase, im, imBG, BG_normalized, imBaseBG
    print("R/G vs R-G")
    imRed = im[:, :, 0]
    imGreen = im[:, :, 1]
    BG_red = imBG[:, :, 0]
    BG_green = imBG[:, :, 1]

    imRed_adjusted = imadjust(imRed, np.min(imRed), np.max(imRed), 0, 255, 1)
    imGreen_adjusted = imadjust(imGreen, np.min(imGreen), np.max(imGreen), 0, 255, 1)
    BG_red_adjusted = imadjust(BG_red, np.min(BG_red), np.max(BG_red), 0, 255, 1)
    BG_green_adjusted = imadjust(BG_green, np.min(BG_green), np.max(BG_green), 0, 255, 1)

    deltaRed = cv2.subtract(imRed, BG_red)
    deltaGreen = cv2.subtract(imGreen, BG_green)

    # deltaRed_norm = cv2.divide(deltaRed, BG_red_adjusted)
    # deltaGreen_norm = cv2.divide(deltaGreen, BG_green_adjusted)
    # deltaRed_norm_adjusted = imadjust(deltaRed_norm, np.min(deltaRed_norm), np.max(deltaRed_norm), 0, 255, 1)
    # deltaGreen_norm_adjusted = imadjust(deltaGreen_norm, np.min(deltaGreen_norm), np.max(deltaGreen_norm), 0, 255, 1)
    DIFFERENCE = cv2.subtract(deltaRed, deltaGreen)
    Ratio = cv2.divide(deltaRed, deltaGreen)
    # DIFFERENCE = cv2.subtract(deltaRed_norm_adjusted, deltaGreen_norm_adjusted)
    # Ratio = cv2.divide(deltaRed_norm_adjusted, deltaGreen_norm_adjusted)

    BG_red_adj = imadjust(BG_red, np.min(BG_red), np.max(BG_red), 0, 255, 1)
    BG_green_adj = imadjust(BG_green, np.min(BG_green), np.max(BG_green), 0, 255, 1)
    DIFFERENCE_adjusted = imadjust(DIFFERENCE, np.min(DIFFERENCE), np.max(DIFFERENCE), 0, 255, 1)
    Ratio_adjusted = imadjust(Ratio, np.min(Ratio), np.max(Ratio), 0, 255, 1)
    deltaRed_adjusted = imadjust(deltaRed, np.min(deltaRed), np.max(deltaRed), 0, 255, 1)
    deltaGreen_adjusted = imadjust(deltaGreen, np.min(deltaGreen), np.max(deltaGreen), 0, 255, 1)



    outputFilename = fileName[:-4] + "_output.png"
    outputFilename_demo = outputFilename[:-4] + "_demo.png"

    coordinate = int(text5.get(1.0, END))

    if Position_Type.get() == 1:  # Vertical line
        if Color_channel.get() == 0:  # Red channel intended
            line_deltaRed = deltaRed[:, coordinate]
            line_deltaGreen = deltaGreen[:, coordinate]
            # line_deltaRed_norm_adjusted = deltaRed_norm_adjusted[:, coordinate]
            # line_deltaGreen_norm_adjusted = deltaGreen_norm_adjusted[:, coordinate]
            line_DIFFERENCE = DIFFERENCE[:, coordinate]
            line_Ratio = Ratio[:, coordinate]
            line_DIFFERENCE_adjusted = DIFFERENCE_adjusted[:, coordinate]
            line_Ratio_adjusted = Ratio_adjusted[:, coordinate]

    elif Position_Type.get() == 0:  # Horizontal line
            line_deltaRed = deltaRed[coordinate, :]
            line_deltaGreen = deltaGreen[coordinate, :]
            # line_deltaRed_norm_adjusted = deltaRed_norm_adjusted[coordinate, :]
            # line_deltaGreen_norm_adjusted = deltaGreen_norm_adjusted[coordinate, :]
            line_DIFFERENCE = DIFFERENCE[coordinate, :]
            line_Ratio = Ratio[coordinate, :]
            line_DIFFERENCE_adjusted = DIFFERENCE_adjusted[coordinate, :]
            line_Ratio_adjusted = Ratio_adjusted[coordinate, :]


    fig1 = plt.figure()
    ax1 = fig1.add_subplot(2, 1, 1)
    ax2 = fig1.add_subplot(2, 1, 2, sharex=ax1)
    ax1.set(title='a)     Fluorescence image')
    ax2.set(title='b)     Plots of modes', xlabel='Pixels', ylabel='Pixels Values', xlim=[0, len(line_deltaRed)])
    ax2.plot(line_deltaRed, color='red', label='red')
    ax2.plot(line_deltaGreen, color='green', label = 'green')
    ax2.plot(line_Ratio, color='black', label='R/G')
    ax2.plot(line_DIFFERENCE, color='blue', label='R-G')
    plt.legend()

    if Position_Type.get() == 1:  # Vertical line
        im[:, coordinate] = 255
    elif Position_Type.get() == 0:  # Horizontal line
        im[coordinate, :] = 255

    ax1.imshow(im)
    plt.savefig(outputFilename_demo)
    plt.show()

    fig2 = plt.figure()
    ax_1 = fig2.add_subplot(2, 6, 1)
    ax_2 = fig2.add_subplot(2, 6, 2)
    ax_3 = fig2.add_subplot(2, 6, 3)
    ax_4 = fig2.add_subplot(2, 6, 4)
    ax_5 = fig2.add_subplot(2, 6, 5)
    ax_6 = fig2.add_subplot(2, 6, 6)
    ax_7 = fig2.add_subplot(2, 6, 7)
    ax_8 = fig2.add_subplot(2, 6, 8)
    ax_9 = fig2.add_subplot(2, 6, 9)
    ax_10 = fig2.add_subplot(2, 6, 10)
    ax_11 = fig2.add_subplot(2, 6, 11)
    ax_12 = fig2.add_subplot(2, 6, 12)
    subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)

    ax_1.set(title='a)     deltaR')
    ax_2.set(title='c)     deltaG')
    ax_3.set(title='e)     R/G')
    ax_4.set(title='g)     R-G')
    ax_5.set(title='i)     R/G adj')
    ax_6.set(title='k)     R-G adj')
    ax_7.set(title='b', xlabel='Pixels', ylabel='Pixels Values', xlim=[0, len(line_deltaRed)])
    ax_8.set(title='d', xlabel='Pixels', xlim=[0, len(line_deltaRed)])
    ax_9.set(title='f', xlabel='Pixels', xlim=[0, len(line_deltaRed)])
    ax_10.set(title='h', xlabel='Pixels', xlim=[0, len(line_deltaRed)])
    ax_11.set(title='j', xlabel='Pixels', xlim=[0, len(line_deltaRed)])
    ax_12.set(title='l', xlabel='Pixels', xlim=[0, len(line_deltaRed)])


    ax_7.plot(line_deltaRed)
    ax_8.plot(line_deltaGreen)
    ax_9.plot(line_Ratio)
    ax_10.plot(line_DIFFERENCE)
    ax_11.plot(line_Ratio_adjusted)
    ax_12.plot(line_DIFFERENCE_adjusted)

    if Position_Type.get() == 1:  # Vertical line
        deltaRed[:, coordinate] = 255
        deltaGreen[:, coordinate] = 255
        # deltaRed_norm_adjusted[:, coordinate] = 255
        # deltaGreen_norm_adjusted[:, coordinate] = 255
        DIFFERENCE[:, coordinate] = 255
        Ratio[:, coordinate] = 255
        DIFFERENCE_adjusted[:, coordinate] = 255
        Ratio_adjusted[:, coordinate] = 255
    elif Position_Type.get() == 0:  # Horizontal line
        deltaRed[coordinate, :] = 255
        deltaGreen[coordinate, :] = 255
        # deltaRed_norm_adjusted[coordinate, :] = 255
        # deltaGreen_norm_adjusted[coordinate, :] = 255
        DIFFERENCE[coordinate, :] = 255
        Ratio[coordinate, :] = 255
        DIFFERENCE_adjusted[coordinate, :] = 255
        Ratio_adjusted[coordinate, :] = 255

    ax_1.imshow(deltaRed, cmap='gray')
    ax_2.imshow(deltaGreen, cmap='gray')
    ax_3.imshow(Ratio, cmap='gray')
    ax_4.imshow(DIFFERENCE, cmap='gray')
    ax_5.imshow(Ratio_adjusted, cmap='gray')
    ax_6.imshow(DIFFERENCE_adjusted, cmap='gray')

    plt.savefig(outputFilename)
    plt.show()

    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    window = Tk()
    window.geometry('1350x250')
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
    btn12 = Button(window, text="R/G", command=R2G)
    btn12.grid(column=2, row=6, sticky=W)
    btn13 = Button(window, text="R-G", command=R_G)
    btn13.grid(column=2, row=7, sticky=W)
    btn14 = Button(window, text="R/G vs R-G", command=comparison)
    btn14.grid(column=2, row=8, sticky=W)

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
