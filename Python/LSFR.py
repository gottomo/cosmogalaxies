# ------------------------------------------------------------------------------
# Abie Marshall
# LSFR.py is a Least Squares Fit Routine with a graphical user interface
# 2016
#
# Adapted from LSFR.m credits to:
# Jordan Hulme
# Adam Petrus
# Ian Duerdoth
# Paddy Leahy
# ------------------------------------------------------------------------------

# Things to do -----------------------------------------------------------------
# TODO add higher order polynomials if required
# TODO rewrite in python 2.7
# TODO check that points are within the specified axis limit
# TODO ability to read csv and other file types
# TODO rewrite the numpy function so less points can be plotted
# TODO make windows and mac interfaces
# TODO maybe rewrite in wxpython or pyQT
# Imports ----------------------------------------------------------------------

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox
from tkinter import filedialog


# TODO check import if doesnt run

import matplotlib
from matplotlib import pyplot as plt

import numpy as np


# Global Variables -------------------------------------------------------------

fitsTypes = ["Linear: y=mx+c", "Quadratic: y=ax^2+bx+c",
             "Cubic: y=ax^3+bx^2+cx+d"]

x = []      # x values
y = []      # y values
e = []      # Error on each point
w = []      # Weight of each point

fittedStructure = 0         # structure of arrays created by polyfit
order = 0                   # Order of fitted polynomial
numberOfParameters = 0      # number of fitted parameters
degreesOfFreedom = 0        # number of degrees of freedom
chisqrd = 0                 # chi squared
redchisqrd = 0              # red chi squared
pError = 0                  # errors on fitted parameters
chisqrdlimit = 0            # max and min limits on chi squared

xmin = float    # holds xmin
xmax = float    # holds xmax
ymin = float    # holds ymin
ymax = float    # holds ymax

filename = str    # directory location of the file
file = str

# These values are set to 1 if an error is encountered in the functions. Stops
# further functions executing and failing
errorValueRead = 0
errorValueFit = 0

# Styles preferences ------------------------------------------------------------

fontStyle = ('times', 12, 'normal')
# -------------------------------------------------------------------------------

# Global Functions --------------------------------------------------------------

# Reading in the data functions


def openDisplaydir(self):

    global filename
    global file
    global errorValueRead

    file = tk.filedialog.askopenfile(filetypes=[("Text files", "*.txt")])

    if file != None:
        filename = file.name

        # Displays Directory ------------------------------------------------
        self.directoryDisplay.configure(state='normal')
        self.directoryDisplay.delete(0, 'end')
        self.directoryDisplay.insert(0, string=filename)
        self.directoryDisplay.configure(state='readonly')

    else:
        errorValueRead = 1


def loadData(self):

    self.filePreview.configure(state='normal')
    self.filePreview.delete(0, 'end')

    try:
        i = 1
        for line in file:
            data = line.split()
            x.append(float(data[0]))
            y.append(float(data[1]))
            e.append(float(data[2]))
            preview = ('%s, %s, %s' % (data[0], data[1], data[2]))
            self.filePreview.insert(i, preview)
            i += 1
        file.close()

    except:
        if filename != 0:
            self.filePreview.delete(0, 'end')
            tk.messagebox.showinfo('Error', (
                    'Cannot load data from\n'
                    '%s\n'
                    '\n'
                    'The data should be a plain text file containing '
                    'three columns of values representing the abscissae, '
                    'ordinates, and error estimates' % filename))

        x[:] = []

        global errorValueRead
        errorValueRead = 1


def checkError(self):
    global errorValueRead

    if min(e) > 0:
        for i in e:
            w.append(1/i)
    else:
        x[:] = []
        self.filePreview.delete(0, 'end')
        tk.messagebox.showinfo('Error', (
                'Errors must be non zero positive values'))
        errorValueRead = 1


def axisEntries(self):
    # Clears the axis entries -------------------------------------------
    self.xMaxEntry.configure(state='normal')
    self.xMinEntry.configure(state='normal')
    self.yMaxEntry.configure(state='normal')
    self.yMinEntry.configure(state='normal')

    self.xMinEntry.delete(0, 'end')
    self.xMaxEntry.delete(0, 'end')
    self.yMinEntry.delete(0, 'end')
    self.yMaxEntry.delete(0, 'end')
    # -------------------------------------------------------------------

    self.numberOfPointsDisplay.config(text=len(x))

    # Set and display auto limits for plot window ------------------------
    plt.plot(x, y)
    global xmin
    global xmax
    global ymin
    global ymax

    [xmin, xmax] = plt.gca().get_xlim()
    [ymin, ymax] = plt.gca().get_ylim()

    self.xMinEntry.insert(0, string=xmin)
    self.xMaxEntry.insert(0, string=xmax)
    self.yMinEntry.insert(0, string=ymin)
    self.yMaxEntry.insert(0, string=ymax)

    self.xMaxEntry.configure(state='disable')
    self.xMinEntry.configure(state='disable')
    self.yMaxEntry.configure(state='disable')
    self.yMinEntry.configure(state='disable')

    plt.clf()

# Plotting Functions ------------------------------------------------------------


def checknumberofpoints(self):
    global numberOfParameters
    global errorValueFit
    errorValueFit = 0                       # resets the errorFitValue value to 0
    plt.clf()
    if self.option.get() == fitsTypes[0]:
        numberOfParameters = 2

    if self.option.get() == fitsTypes[1]:
        numberOfParameters = 3

    if self.option.get() == fitsTypes[2]:
        numberOfParameters = 4


def figurelabel(self):

    if self.residualsCheckCmd.get() == 1 and self.parametersCheckCmd.get() == 1:
            plt.subplot(2, 1, 1)

    elif self.residualsCheckCmd.get() == 1 and self.parametersCheckCmd.get() == 0:
            plt.subplot(2, 1, 1)

    elif self.residualsCheckCmd.get() == 0 and self.parametersCheckCmd.get() == 1:
                plt.subplot(1, 1, 1)
                plt.gca().set_position((.1, .3, .8, .6))
    else:
        plt.subplot(1, 1, 1)

    plt.title(self.plotTitleEntry.get())
    plt.xlabel(self.xLabelEntry.get())
    plt.ylabel(self.yLabelEntry.get())

    if self.gridLineCheckCmd.get() == 1:
        plt.grid(True)


def limits(self):

    global xmin
    global xmax
    global ymin
    global ymax
    global errorValueFit

    try:
        if self.xMinAutoCheckCmd.get() == 0:
            xmin = float(self.xMinEntry.get())
            if xmin < xmax:
                plt.xlim(xmin=xmin, xmax=xmax)
            else:
                errorValueFit = 1
                tk.messagebox.showinfo('Error', (
                'xmin must be less than xmax'
            ))
        if self.xMaxAutoCheckCmd.get() == 0:
            xmax = float(self.xMaxEntry.get())
            if xmin < xmax:
                plt.xlim(xmin=xmin, xmax=xmax)
            else:
                errorValueFit = 1
                tk.messagebox.showinfo('Error', (
                'xmin must be less than xmax'
            ))
        if self.yMinAutoCheckCmd.get() == 0:
            ymin = float(self.yMinEntry.get())
            if ymin < ymax:
                plt.ylim(ymin=ymin, ymax=ymax)
            else:
                errorValueFit = 1
                tk.messagebox.showinfo('Error', (
                'ymin must be less than ymax'
            ))
        if self.yMaxAutoCheckCmd.get() == 0:
            ymax = float(self.yMaxEntry.get())
            if ymin < ymax:
                plt.ylim(ymin=ymin, ymax=ymax)
            else:
                errorValueFit = 1
                tk.messagebox.showinfo('Error', (
                'ymin must be less than ymax'
            ))
    except:
        tk.messagebox.showinfo('Error', (
                'Invalid character in axis entry field, '
                'entered values should be numbers either '
                'typed explicitly or expressed in scientific notation.\n'
                'ie, 125 or 1.25e2 '
            ))
        errorValueFit = 1


def fitting(self):
    global degreesOfFreedom
    global fittedStructure
    global order
    global numberOfParameters
    global errorValueFit

    if self.option.get() == fitsTypes[0]:
        order = 1
        fittedStructure = np.polyfit(x, y, order, cov=True, w=w)
        degreesOfFreedom = len(x)-numberOfParameters
        plt.annotate('Fit: $y=mx+c$', (0, 0), (0, -20),
                 xycoords='axes fraction', textcoords='offset points',
                 va='top')

    if self.option.get() == fitsTypes[1]:
        order = 2
        fittedStructure = np.polyfit(x, y, order, cov=True)
        degreesOfFreedom = len(x)-numberOfParameters
        plt.annotate('Fit: $y=ax^2+bx+c$', (0, 0), (0, -20),
                 xycoords='axes fraction', textcoords='offset points',
                 va='top')

    if self.option.get() == fitsTypes[2]:
        order = 3
        fittedStructure = np.polyfit(x, y, order, cov=True)
        degreesOfFreedom = len(x)-numberOfParameters
        plt.annotate('Fit: $y=ax^3+bx^2+cx+d$', (0, 0), (0, -20),
                 xycoords='axes fraction', textcoords='offset points',
                 va='top')


def errorinformation(self):

    global degreesOfFreedom
    global fittedStructure
    global pError
    global chisqrd
    global redchisqrd
    global chisqrdlimit

    df = degreesOfFreedom
    p = fittedStructure[0]

    chisqrd = 0

    for i, j, k in zip(x, y, e):
        c = pow(((j - np.polyval(p, i))/k), 2)
        chisqrd += c

    redchisqrd = chisqrd/df

    # note the scaling factor of * (len(x) - numberOfParameters - 2)/chisqrd.
    # This is to account for an offset introduced by numpy.polyfit
    cov = fittedStructure[1] * (len(x) - numberOfParameters - 2)/chisqrd

    pError = np.sqrt(np.diag(cov))      # diagonals of covariance matrix

    # chi squared limit --------------------------------------------------
    if df < 20:
        chisqrdlimit = [2, 0.5]
    else:
        chisqrdlimit = [1+np.sqrt(8/df), 1-np.sqrt(8/df)]


def plotannotate(self):
    # import variables ----------------------------------------------------
    p = fittedStructure[0]
    # ----------------------------------------------------------------------
    # Plot graph and fitted line -------------------------------------------
    plt.errorbar(x, y, e, linestyle='none', color='k')
    [xmin, xmax]=plt.gca().get_xlim()
    xlim = np.linspace(xmin, xmax, 500)

    if redchisqrd <= chisqrdlimit[0] and redchisqrd >= chisqrdlimit[1]:
        plt.plot(xlim, np.polyval(p, xlim), 'b')
    else:
        plt.plot(xlim, np.polyval(p, xlim), 'r')
        tk.messagebox.showinfo('Warning', (
                'The Fitted line does not well '
                'describe the data. For this data '
                'the acceptable range of chi squared '
                'is between %4.2f - %4.2f'
                %(chisqrdlimit[1], chisqrdlimit[0])
            ))
    # ----------------------------------------------------------------------
    if self.parametersCheckCmd.get() == 1:
        # Displays chi values --------------------------------------------------
        plt.annotate((r'$\chi^2$ = %4.2f' %(chisqrd)), (1, 0), (-60, -30),
                     xycoords='axes fraction', textcoords='offset points',
                     va='top')

        plt.annotate(('Degrees of Freedom = %s' %(degreesOfFreedom)), (1, 0), (-150, -50),
                     xycoords='axes fraction', textcoords='offset points',
                     va='top')

        plt.annotate((r'reduced $\chi^2$ = %4.2f' %(redchisqrd)), (1, 0), (-104, -70),
                     xycoords='axes fraction', textcoords='offset points',
                     va='top')
        # ----------------------------------------------------------------------

        # Display fitted parameters along with with errors ---------------------
        # TODO compress this section using some loops --------------------------
        # Linear ---------------------------------------------------------------
        if self.option.get() == fitsTypes[0]:
            plt.annotate(('m = %6.4e' % (p[0])), (0, 0), (0, -40),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[0])), (0, 0), (100, -40),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('c = %6.4e' % (p[1])), (0, 0), (0, -55),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[1])), (0, 0), (100, -55),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
        # -----------------------------------------------------------------------

        # Quadratic -------------------------------------------------------------
        elif self.option.get() == fitsTypes[1]:
            plt.annotate(('a = %6.4e' % (p[0])), (0, 0), (0, -40),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[0])), (0, 0), (100, -40),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('b = %6.4e' % (p[1])), (0, 0), (0, -55),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[1])), (0, 0), (100, -55),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('c = %6.4e' % (p[2])), (0, 0), (0, -70),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[2])), (0, 0), (100, -70),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
        # -----------------------------------------------------------------------

        # Cubic -----------------------------------------------------------------
        elif self.option.get() == fitsTypes[2]:
            plt.annotate(('a = %6.4e' % (p[0])), (0, 0), (0, -40),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[0])), (0, 0), (100, -40),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('b = %6.4e' % (p[1])), (0, 0), (0, -55),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[1])), (0, 0), (100, -55),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('c = %6.4e' % (p[2])), (0, 0), (0, -70),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[2])), (0, 0), (100, -70),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('d = %6.4e' % (p[3])), (0, 0), (0, -85),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')
            plt.annotate(('± %6.4e' % (pError[3])), (0, 0), (100, -85),
                         xycoords='axes fraction', textcoords='offset points',
                         va='top', fontsize='10')


def residuals(self):
    global xmin
    global xmax

    p = fittedStructure[0]
    if self.residualsCheckCmd.get() == 1:
        plt.subplot(4, 1, 4)
        plt.title('Residuals')

        fitted = np.polyval(p, x) - y
        plt.errorbar(x, fitted, e, linestyle='none', color='k', marker='x')
        plt.locator_params(axis='y', nbins=2)   #sets the y axis ticks

        if self.xMinAutoCheckCmd.get() == 0:
            plt.xlim(xmin=xmin, xmax=xmax)
        if self.xMaxAutoCheckCmd.get() == 0:
            plt.xlim(xmin=xmin, xmax=xmax)
        else:
            [xmin, xmax]=plt.gca().get_xlim()
            plt.xlim(xmin=xmin, xmax=xmax)

        xlim = np.linspace(xmin, xmax, 500)
        zeros = 0*xlim

        # checks chi squared value
        if redchisqrd <= chisqrdlimit[0] and redchisqrd >= chisqrdlimit[1]:
            plt.plot(xlim, zeros, marker='', color='b')
        else:
            plt.plot(xlim, zeros, marker='', color='r')
# -------------------------------------------------------------------------------

# Main Menu ---------------------------------------------------------------------

class MainMenu:
    def __init__(self,master):

        # Graphical User Interface ----------------------------------------------

        self.s = ttk.Style()
        self.s.configure('TLabel', foreground='Black', background='white')
        self.s.configure('TButton', foreground='black', background='white')
        self.s.configure('about.TLabel', forground='black', background='#f2d211')
        self.s.configure('TEntry', foreground='Blue')
        self.s.configure('TCheckbutton', foreground='black', background='white')
        self.s.configure('TMenubutton', background='white', foreground='black', width=25)
        self.s.configure('TLabelframe', foreground='black', background='white')
        self.s.configure('TFrame', background='white')

        # Window properties -----------------------------------------------------
        self.master = master
        master.wm_title("LSFR for Python")
        master.resizable(0, 0)
        # -----------------------------------------------------------------------

        # Variables -------------------------------------------------------------
        self.option = tk.StringVar(master)    # default fit type
        self.option.set(fitsTypes[0])         # default fit type

        self.directoryString = tk.StringVar(master)
        self.fileNameString = tk.StringVar(master)
        self.numberOfPointsString = tk.StringVar(master)

        self.gridLineCheckCmd = tk.IntVar()     # Gridline check command
        self.gridLineCheckCmd.set(1)

        self.residualsCheckCmd = tk.IntVar()    # resiuduals check command
        self.residualsCheckCmd.set(1)

        self.parametersCheckCmd = tk.IntVar()   # parametres check command
        self.parametersCheckCmd.set(1)

        self.xMinAutoCheckCmd = tk.IntVar()     # X min check command
        self.xMinAutoCheckCmd.set(1)

        self.xMaxAutoCheckCmd = tk.IntVar()     # X max check command
        self.xMaxAutoCheckCmd.set(1)

        self.yMinAutoCheckCmd = tk.IntVar()     # Y min check command
        self.yMinAutoCheckCmd.set(1)

        self.yMaxAutoCheckCmd = tk.IntVar()     # Y max check command
        self.yMaxAutoCheckCmd.set(1)
        # -----------------------------------------------------------------------
        # Frames ----------------------------------------------------------------
        self.frame0 = tk.Frame(master, bd=2, bg='white')
        self.frame1 = tk.LabelFrame(master, text=" Plot Options ", background='white')
        self.xAxisLimits = tk.LabelFrame(self.frame1, text=" X-Axis Limits ", bg='white')
        self.yAxisLimits = tk.LabelFrame(self.frame1, text=" Y-Axis Limits ", background='white')

        self.frame2 = tk.LabelFrame(master, text=" Data Options ", background='white')

        self.frame3 = tk.LabelFrame(master, text=" File Preview ", background='white')

        self.frame4 = ttk.Frame(master)
        # -----------------------------------------------------------------------

        # Frame grid positions --------------------------------------------------
        self.frame0.grid(row=0, columnspan=2, sticky='NSEW')

        self.frame1.grid(row=1, columnspan=2, sticky='NSEW',
                         padx=5, pady=5, ipadx=5, ipady=15)
        self.xAxisLimits.grid(row=0, column=3,
                              padx=5, pady=5, ipadx=5, ipady=5, sticky='E')

        self.yAxisLimits.grid(row=1, rowspan=3, column=3,
                              padx=5, pady=5, ipadx=5, ipady=5)

        self.frame2.grid(row=2, rowspan=3, column=0, sticky='NESW',
                         padx=5, pady=5, ipadx=1, ipady=5)

        self.frame3.grid(row=2, rowspan=5, column=1, sticky='NESW',
                         padx=5, pady=5, ipadx=0, ipady=5)

        self.frame4.grid(row=6, column=0, sticky='NESW',
                         padx=5, pady=5, ipadx=0, ipady=5)
        # -----------------------------------------------------------------------

        # Frame 0 widgets  ------------------------------------------------------
        self.about = ttk.Button(self.frame0, text=" About LSFR ")

        self.about.bind("<Button-1>", self.aboutLSFR)
        # -----------------------------------------------------------------------

        # Frame 0 widgets positions  --------------------------------------------
        self.about.grid(row=0, padx=7)
        # -----------------------------------------------------------------------
        # Frame 1 widgets  ------------------------------------------------------
        self.fitType = ttk.Label(self.frame1, text="Fit type: ")
        self.plotTitle = ttk.Label(self.frame1, text="Plot title: ")
        self.xLabel = ttk.Label(self.frame1, text="X-axis label: ")
        self.yLabel = ttk.Label(self.frame1, text="Y-axis label: ")

        self.fitTypeChoice = ttk.OptionMenu(
            self.frame1, self.option,
            '', fitsTypes[0], fitsTypes[1], fitsTypes[2])

        self.plotTitleEntry = ttk.Entry(self.frame1)
        self.xLabelEntry = ttk.Entry(self.frame1)
        self.yLabelEntry = ttk.Entry(self.frame1)

        self.plotTitleEntry.insert(0, string='Plot of my Data')
        self.xLabelEntry.insert(0, string='Abscissa')
        self.yLabelEntry.insert(0, string='Ordinate')

        self.gridLinesCheck = ttk.Checkbutton(self.frame1,
                text='Grid lines', variable=self.gridLineCheckCmd, onvalue=1, offvalue=0)

        self.residualsCheck = ttk.Checkbutton(self.frame1,
                text= 'Show residual plot', variable=self.residualsCheckCmd, onvalue=1, offvalue=0)

        self.showParametersCheck = ttk.Checkbutton(self.frame1,
                text='Show fitted Parameters', variable=self.parametersCheckCmd, onvalue=1, offvalue=0)

        self.frame1Spacer = ttk.Label(self.frame1, text="                                                       ")
        # -----------------------------------------------------------------------

        # Frame 1 widgets positions  --------------------------------------------
        self.fitType.grid(row=0, column=0, pady=30, sticky='E')
        self.fitTypeChoice.grid(row=0, column=1, sticky='EW')

        self.plotTitle.grid(row=1, column=0, sticky='E')
        self.plotTitleEntry.grid(row=1, column=1, sticky='W')

        self.xLabel.grid(row=2, column=0, sticky='E')
        self.xLabelEntry.grid(row=2, column=1, sticky='W')

        self.yLabel.grid(row=3, column=0, sticky='E')
        self.yLabelEntry.grid(row=3, column=1, sticky='W')

        self.gridLinesCheck.grid(row=4, column=1, sticky='W')

        self.residualsCheck.grid(row=5, column=1, sticky='W')

        self.showParametersCheck.grid(row=6, column=1, sticky='W')

        self.frame1Spacer.grid(row=0,column=2)
        # -----------------------------------------------------------------------

        # X-Axis Limit widgets  -------------------------------------------------
        self.xMin = ttk.Label(self.xAxisLimits, text='Min')
        self.xMinEntry = ttk.Entry(self.xAxisLimits, width=7, state='disabled')
        self.xMinAutoCheck = ttk.Checkbutton(
            self.xAxisLimits, variable=self.xMinAutoCheckCmd, onvalue=1, offvalue=0,
            command=self.disablexMinEntry, text='Auto')

        self.xMax = ttk.Label(self.xAxisLimits, text='Max')
        self.xMaxEntry = ttk.Entry(self.xAxisLimits, width=7, state='disabled')
        self.xMaxAutoCheck = ttk.Checkbutton(
            self.xAxisLimits, variable=self.xMaxAutoCheckCmd, onvalue=1, offvalue=0,
            command=self.disablexMaxEntry, text='Auto')

        # -----------------------------------------------------------------------

        # X-Axis Limit widgets positions ----------------------------------------
        self.xMin.grid(row=0, column=0, sticky='E', pady=5)
        self.xMinEntry.grid(row=0, column=1, sticky='W', pady=5)
        self.xMinAutoCheck.grid(row=0, column=2, padx=1, ipadx=1, pady=5)

        self.xMax.grid(row=1, column=0, sticky='E', pady=5)
        self.xMaxEntry.grid(row=1, column=1, sticky='W', pady=5)
        self.xMaxAutoCheck.grid(row=1, column=2, padx=1, ipadx=1, pady=5)
        # -----------------------------------------------------------------------

        # Y-Axis Limit widgets  -------------------------------------------------
        self.yMin = ttk.Label(self.yAxisLimits, text='Min')
        self.yMinEntry = ttk.Entry(self.yAxisLimits, width=7, state='disable')
        self.yMinAutoCheck = ttk.Checkbutton(
            self.yAxisLimits, variable=self.yMinAutoCheckCmd, onvalue=1, offvalue=0,
            command=self.disableyMinEntry, text="Auto")

        self.yMax = ttk.Label(self.yAxisLimits, text='Max')
        self.yMaxEntry = ttk.Entry(self.yAxisLimits, width=7, state='disable')
        self.yMaxAutoCheck = ttk.Checkbutton(
            self.yAxisLimits, variable=self.yMaxAutoCheckCmd, onvalue=1, offvalue=0,
            command=self.disableyMaxEntry, text="Auto")
        # -----------------------------------------------------------------------

        # Y-Axis Limit widgets positions ----------------------------------------
        self.yMin.grid(row=0, column=0, pady=5)
        self.yMinEntry.grid(row=0, column=1,pady=5)
        self.yMinAutoCheck.grid(row=0, column=2, padx=1, ipadx=1, pady=5)

        self.yMax.grid(row=1, column=0, pady=5)
        self.yMaxEntry.grid(row=1, column=1,pady=5)
        self.yMaxAutoCheck.grid(row=1, column=2, padx=1, ipadx=1, pady=5)
        # -----------------------------------------------------------------------

        # Frame 2 widgets  ------------------------------------------------------
        self.loadData = ttk.Button(self.frame2, text=" Load new data... ",
                                  command=self.openfile)
        self.directory = ttk.Label(self.frame2, text="Working directory: ")
        self.directoryDisplay = ttk.Entry(self.frame2, width=27)
        self.directoryDisplay.insert(0, string='(no directory selected)')
        self.directoryDisplay.configure(state='readonly')

        self.numberOfPoints = ttk.Label(self.frame2, text="Number of data points: ")
        self.numberOfPointsDisplay = ttk.Label(self.frame2, text="0")
        # -----------------------------------------------------------------------

        # Frame 2 widgets positions  --------------------------------------------
        self.directory.grid(row=0, column=0, sticky='E', padx=5, pady=1)
        self.directoryDisplay.grid(row=0, column=1, sticky='WE', padx=1, pady=1)

        self.numberOfPoints.grid(row=1, column=0, sticky='E', padx=5, pady=1)
        self.numberOfPointsDisplay.grid(row=1, column=1, sticky='W', padx=5, pady=1)

        self.loadData.grid(row=2, column=0, sticky='NESW', padx=5, pady=1, columnspan=2)

        # -----------------------------------------------------------------------

        # Frame 3 widgets  ------------------------------------------------------
        self.filePreview = tk.Listbox(self.frame3, borderwidth=0, width=40)
        self.filePreview.insert(1,'(no file loaded)')
        self.filePreview.configure(state='disable')
        # -----------------------------------------------------------------------

        # Frame 3 widgets positions  --------------------------------------------
        self.filePreview.pack(fill='both')
        # -----------------------------------------------------------------------

        # Frame 4 widgets  ------------------------------------------------------
        self.exit = ttk.Button(self.frame4, text=' Exit ', command=self.destroy, width=4)
        self.plot = ttk.Button(self.frame4, text=" Plot ", command=self.plot, width=6)
        self.help = ttk.Button(self.frame4, text=" Help ", command=self.helpwindow, width=4)
        # -----------------------------------------------------------------------

        # Frame 4 widgets positions  --------------------------------------------
        self.exit.grid(row=0, column=0, sticky='EW',
                       padx=30, pady=5, ipadx=5, ipady=5)

        self.plot.grid(row=0, column=1, sticky='EW',
                       padx=30, pady=5, ipadx=5, ipady=5)

        self.help.grid(row=0, column=2, sticky='EW',
                       padx=30, pady=5, ipadx=5, ipady=5)
        # -----------------------------------------------------------------------

    # Main Menu Button Functions ------------------------------------------------
    # Makes the axis entries disabled/normal depending on auto checkCmd ---------
    def disablexMinEntry(self):
        if self.xMinAutoCheckCmd.get() == 1:
            self.xMinEntry.configure(state='disabled')
            self.xMinAutoCheckCmd.set(1)
        else:
            self.xMinEntry.configure(state='normal')
            self.xMinAutoCheckCmd.set(0)

    def disablexMaxEntry(self):
        if self.xMaxAutoCheckCmd.get() == 1:
            self.xMaxEntry.configure(state='disabled')
            self.xMaxAutoCheckCmd.set(1)
        else:
            self.xMaxEntry.configure(state='normal')
            self.xMaxAutoCheckCmd.set(0)

    def disableyMinEntry(self):
        if self.yMinAutoCheckCmd.get() == 1:
            self.yMinEntry.configure(state='disable')
            self.yMinAutoCheckCmd.set(1)
        else:
            self.yMinEntry.configure(state='normal')
            self.yMinAutoCheckCmd.set(0)

    def disableyMaxEntry(self):
        if self.yMaxAutoCheckCmd.get() == 1:
            self.yMaxEntry.configure(state='disable')
            self.yMaxAutoCheckCmd.set(1)
        else:
            self.yMaxEntry.configure(state='normal')
            self.yMaxAutoCheckCmd.set(0)

    # Close the application -----------------------------------------------------
    def destroy(self):
        self.master.quit()
        self.master.destroy()

    # Select file to open -------------------------------------------------------
    def openfile(self):
        global errorValueRead
        errorValueRead = 0
        # Clears any previous data ------------------------------------------
        x[:] = []
        y[:] = []
        e[:] = []
        w[:] = []

        if errorValueRead == 0:
            openDisplaydir(self)    # reads in the data, displays preview and directory
        if errorValueRead == 0:
            loadData(self)
        if errorValueRead == 0:
            checkError(self)
        if errorValueRead == 0:
            axisEntries(self)       # sets the automatic axis entries

    # Creates and displays plot window ----------------------------------------
    # -------------------------------------------------------------------------
    def plot(self):
        checknumberofpoints(self)
        if len(x) != 0 and (len(x) - numberOfParameters - 2) <= 0:
            tk.messagebox.showinfo('Error', (
                'More data points are needed for this fit'
            ))

        elif len(x) == 0:
            tk.messagebox.showinfo('Error', (
                    'No file selected'
                ))

        else:
            if errorValueFit == 0:
                figurelabel(self)       # Creates figure and labels axis of main plot
            if errorValueFit == 0:
                limits(self)            # Limits
            if errorValueFit == 0:
                fitting(self)           # Fits the selected polynomial
            if errorValueFit == 0:
                errorinformation(self)  # Calculates error on fit values
            if errorValueFit == 0:
                plotannotate(self)      # Plots and annotates the graph
            if errorValueFit == 0:
                residuals(self)         # Plots the residuals
            if errorValueFit == 0:
                plt.show()

    # -------------------------------------------------------------------------

    # Top windows -------------------------------------------------------------

    def aboutLSFR(self, event):
        self.aboutWindow = tk.Toplevel(self.master)
        self.app = About(self.aboutWindow)


    def helpwindow(self):
        self.helpWindow = tk.Toplevel(self.master)
        self.app = Window(self.helpWindow)


# About Window ---------------------------------------------------------------

class About:
    def __init__(self, master):

        self.master = master
        master.wm_title("About LSFR")
        master.geometry('300x260')
        master.resizable(0, 0)
        master.configure(background='white')

        # Frames -------------------------------------------------------------
        self.frame1 = ttk.Frame(master)
        self.frame1.pack()

        # Widgets ------------------------------------------------------------

        self.aboutText = tk.Label(self.frame1, bg='white',
                        text='\n'
                             'LSFR (version BETA 1.0)\n'
                             '\n'
                             'A Graph Plotting and Data Analysis\n'
                             'Program for Python\n'
                             '\n'
                             '(c) Abie Marshall 2016\n'
                             'Department of Physics and Astronomy\n'
                             'The University of Manchester\n'
                             '\n'
                             'Adapated from LSFR.m 2003\n'
                             'Jordan Hulme, Adam Petrus, Ian Duerdoth,\n'
                             'Paddy Leahy\n')


        self.close = ttk.Button(self.frame1, text='Close'
                                , width=20, command=self.closeWindow)

        # Positions -------------------------------------- --------------------

        self.aboutText.grid(row=0)

        self.close.grid(row=1)

    def closeWindow(self):
        self.master.destroy()

# Help Window -----------------------------------------------------------------


class Window:
    def __init__(self, master):

        self.master = master
        master.wm_title("Help")
        master.resizable(0, 0)

        self.frame1 = tk.Frame(master)

        self.frame1.pack()

        self.helpText = tk.Text(self.frame1)
        self.helpText.insert('1.0',
            (
                "================================================================================\n"
                "                                  LSFR HELP MENU\n"
                "================================================================================\n"
                "\n"
                "                              (c) Abie Marshall 2016\n"
                "                       abie.marshall@student.manchester.ac.uk\n"
                "                           The University of Manchester\n"
                "                          School Of Physics and Astronomy\n"
                "                     Least Squares Fitting Routine for Python\n"
                "                                 Beta Version 1.0\n"
                "\n"
                "================================================================================\n"
                "                                  FITTING METHOD\n"
                "================================================================================\n"
                "\n"
                "This program fits a weighted least square polynomial to the provided data using\n"
                "numpy.polyfit documentation at: \n"
                "\n"
                "http://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.polyfit.html\n"
                "\n"
                "Errors on the fitted parameters are calculated from the corresponding covariance\n"
                "matrix\n"
                "\n"
                "The minimum number of data points required for this method is:\n"
                "number of fitted parameters + 3\n"
                "ie, for a linear fit at least 5 points are required\n"
                "\n"
                "================================================================================\n"
                "                                  DATA ANALYSIS\n"
                "================================================================================\n"
                "\n"
                "The chi squared indicates how well the fit describes the data. For less than 20 \n"
                "points the reduced chi squared should be between 0.5 and 2. For more points the\n"
                "acceptable range narrows and is given by:\n"
                "reduced chi squared = 1 ± sqrt(8/degreesOfFreedom)\n"
                "\n"
                "For a good fit the residuals should be randomly dispersed about the horizontal\n"
                "axis. If This is not the case the residuals can indicate what type of fit would\n"
                "better described the data"
                "\n"
                "================================================================================\n"
                "                                  PLOT OPTIONS\n"
                "================================================================================\n"
                "\n"
                "Fit type................................................Select fitted polynomial\n"
                "Show grid lines/residuals/parameters.............Show/hide object on plot window\n"
                "Plot title/Axis Labels.................................Labels on the plot window\n"
                "\n"
                "Mathematical and Greek symbols can be typed into the plot labels. See:\n"
                "\n"
                "http://matplotlib.org/users/mathtext.html\n"
                "\n"
                "For a full documentation of available symbols\n"
                "\n"
                "Example:\n"
                "$\ alpha_0$...... Will produce alpha subscript 0 in whichever label it is typed\n"
                "Note there shouldn't be a space between the backslash and the expression when\n"
                "typed into label box (its a formatting thing having to have a space here)\n"
                "\n"
                "================================================================================\n"
                "                                   DATA OPTIONS\n"
                "================================================================================\n"
                "\n"
                "Load new data.............................................Loads a new data file\n"
                "Preview of data will be displayed in the preview window and current directory\n"
                "will be displayed\n"
                "\n"
                "================================================================================\n"
                "                                    BUTTONS\n"
                "================================================================================\n"
                "\n"
                "Plot..................................................Plot with current options\n"
                "Help.......................................................Displays this window\n"
                "Exit...............................................................Exit LSFR.py\n"
            ))

        self.helpText.pack()
        self.helpText.configure(state='disabled')

# Main Root -----------------------------------------------------------------

def main():
    root = tk.Tk()
    app = MainMenu(root)
    app.master.configure(background='white')
    root.mainloop()

if __name__ == '__main__':
    main()

# ---------------------------------------------------------------------------