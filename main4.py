import time
import tkinter as tk
from tkinter import ttk
import tkinter.scrolledtext as scrolledtext
import numpy as np
import pandas as pd
import pyarrow as pa  # just so that pandas gets off my nerves with the constant warnings -.-
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
from multiprocessing import Process
from pympler import tracker


class ListFrame(ttk.Frame):
    def __init__(self, parent, text_data, list_enable, color_var, size_var, item_height):
        super().__init__(master=parent)
        self.pack(expand=True, fill='both')

        # widget data
        self.text_data = text_data
        self.list_enable = list_enable
        self.item_number = len(text_data)
        self.list_height = self.item_number * item_height
        self.color_var = color_var
        self.size_var = size_var

        # canvas
        self.canvas = tk.Canvas(self, background='red', scrollregion=(0, 0, self.winfo_width(), self.list_height))
        self.canvas.pack(expand=True, fill='both')

        # display frame
        self.frame = ttk.Frame(self)

        # creating first row
        frame = ttk.Frame(self.frame)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure((0, 1, 2, 3), weight=1, uniform='a')
        ttk.Label(frame, text='Columns').grid(row=0, column=0, sticky='w')
        ttk.Label(frame, text='Enable').grid(row=0, column=1, sticky='w')
        ttk.Label(frame, text='Colour').grid(row=0, column=2, sticky='w')
        ttk.Label(frame, text='Size').grid(row=0, column=3, sticky='w')
        frame.pack(fill='x', pady=5, padx=5)

        index = 0  # could probably been done more elegantly, but I dont know how ... -> enumerate ...
        for item in self.text_data:
            self.create_item(item, self.list_enable[index], index, self.color_var, self.size_var).pack(fill='x', pady=5,
                                                                                                       padx=2)
            index += 1

        # scrollbar
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.place(relx=1, rely=0, relheight=1, anchor='ne')

        # events
        self.canvas.bind_all('<MouseWheel>', lambda event: self.canvas.yview_scroll(-int(event.delta / 60), "units"))
        self.bind('<Configure>', self.update_size)

    def update_size(self, event):
        # '''
        if self.list_height >= self.winfo_height():
            height = self.list_height
            self.canvas.bind_all('<MouseWheel>',
                                 lambda event: self.canvas.yview_scroll(-int(event.delta / 60), "units"))
            self.scrollbar.place(relx=1, rely=0, relheight=1, anchor='ne')
        else:
            height = self.winfo_height()
            self.canvas.unbind_all('<MouseWheel>')
            self.scrollbar.place_forget()
        # '''

        self.canvas.create_window(
            (0, 0),
            window=self.frame,
            anchor='nw',
            width=self.winfo_width(),
            # height=self.list_height)
            height=height)

    def create_item(self, item, enable, index, color_var, size_var):
        frame = ttk.Frame(self.frame)

        # print('create item', time.time_ns())

        # grid layout
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure((0, 1, 2, 3), weight=1, uniform='a')

        # widgets
        ttk.Label(frame, text=f'{item}').grid(row=0, column=0, sticky='w')
        ttk.Checkbutton(frame, variable=enable, onvalue=True, offvalue=False, takefocus=False, command=command_choose_columns, state='normal').grid(row=0, column=1, sticky='w')
        ttk.Radiobutton(frame, variable=color_var, takefocus=False, value=index, command=quickRedraw).grid(row=0,
                                                                                                           column=2,
                                                                                                           sticky='w')
        ttk.Radiobutton(frame, variable=size_var, takefocus=False, value=index, command=quickRedraw).grid(row=0,
                                                                                                          column=3,
                                                                                                          sticky='w')

        # ttk.Label(frame, text=f'#{index}').grid(row=0, column=0)
        # ttk.Label(frame, text=f'{item[0]}').grid(row=0, column=1)
        # ttk.Button(frame, text=f'{item[1]}').grid(row=0, column=2, columnspan=2, sticky='nsew')

        return frame


# GUI Functions -------------------------------------------------------------------------------------------------------
# the command functions for the radio buttons, chooses the dataset to be used
def radio_command_choseData():
    chooseSettingsForCalculation()


# if changes, also look into check1() as there is code duplication (why? Because it is either this or a more annoying GUI)
def radio_command_dimRed():
    chooseSettingsForCalculation()


# -> the checkbox for the standard scalar -> also computes dim-red for more intuitive gui
# -> meaning it changes the graphic which is an instant response
def check1():
    global enable_standard_scalar
    if check_var1.get() == 1:
        enable_standard_scalar = True
    else:
        enable_standard_scalar = False

    chooseSettingsForCalculation()


# activates when a checkbox in table-selection is pressed -> how to identify which checkbox got pressed???
# -> check with default -> we get the one change -> and then update the default to save the change
def command_choose_columns():
    # warn of selection of color and size for newly selected columns
    global selected_columns_changed
    selected_columns_changed = True

    if radio_data.get() == 1:
        data_chosen_columns = data_1_global.copy()
        for index, item in enumerate(list_enable_default_1):
            if not item == list_enable_data_1[index].get():
                # test if default and gui selection is different -> that one change is what caused the event!
                # -> the index is the column that is different in default and data_1
                list_enable_default_1[index] = list_enable_data_1[index].get()  # adjust so the next change can also be detected

                # compute the total row count BUT consider only the active columns, before and after the new selection
                # cut data_active to only the active columns
                for index2, (series_name, series) in enumerate(data_1_global.items()): # -> drop all columns that are not selected, remove nans and count
                    if not list_enable_data_1[index2].get():
                        data_chosen_columns.drop(series_name, axis=1, inplace=True)

                # def printColumnInfo(data, index, row_count_before_selection, row_count_after_selection):
                printColumnInfo(data_1_global, index, len(data_chosen_columns), len(data_chosen_columns.dropna()))

                # disable / enable the corresponding radio buttons
                # [index+1] -> first row is the row of names
                # [2] -> that is the color radio button of that row
                # list_enable_data_1 holds the new change
                if list_enable_data_1[index].get():  # that is where something changed, no has the new value
                    list_frame_data_1.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'normal'
                    list_frame_data_1.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'normal'
                else:
                    list_frame_data_1.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'disable'
                    list_frame_data_1.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'disable'
                    if index == color_data_1.get():
                        color_data_1.set(data_1_global.columns.get_loc(data_chosen_columns.columns.values[0]))
                    if index == size_data_1.get():
                        size_data_1.set(data_1_global.columns.get_loc(data_chosen_columns.columns.values[0]))

                # this can sadly NOT be exchanged for "data_chosen_columns" at the top ...
                global data_1_selected_columns
                data_1_selected_columns = data_chosen_columns.copy()

    elif radio_data.get() == 2:
        data_chosen_columns = data_2_global.copy()
        for index, item in enumerate(list_enable_default_2):
            if not item == list_enable_data_2[index].get():
                list_enable_default_2[index] = list_enable_data_2[index].get()

                for index2, (series_name, series) in enumerate(data_2_global.items()):
                    if not list_enable_data_2[index2].get():
                        data_chosen_columns.drop(series_name, axis=1, inplace=True)

                printColumnInfo(data_2_global, index, len(data_chosen_columns), len(data_chosen_columns.dropna()))

                if list_enable_data_2[index].get():
                    list_frame_data_2.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'normal'
                    list_frame_data_2.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'normal'
                else:
                    list_frame_data_2.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'disable'
                    list_frame_data_2.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'disable'
                    if index == color_data_2.get():
                        color_data_2.set(data_2_global.columns.get_loc(data_chosen_columns.columns.values[0]))
                    if index == size_data_2.get():
                        size_data_2.set(data_2_global.columns.get_loc(data_chosen_columns.columns.values[0]))

                global data_2_selected_columns
                data_2_selected_columns = data_chosen_columns.copy()

    elif radio_data.get() == 3:
        data_chosen_columns = data_3_global.copy()
        for index, item in enumerate(list_enable_default_3):
            if not item == list_enable_data_3[index].get():
                list_enable_default_3[index] = list_enable_data_3[index].get()

                for index2, (series_name, series) in enumerate(data_3_global.items()):
                    if not list_enable_data_3[index2].get():
                        data_chosen_columns.drop(series_name, axis=1, inplace=True)

                printColumnInfo(data_3_global, index, len(data_chosen_columns), len(data_chosen_columns.dropna()))

                if list_enable_data_3[index].get():
                    list_frame_data_3.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'normal'
                    list_frame_data_3.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'normal'
                else:
                    list_frame_data_3.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'disable'
                    list_frame_data_3.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'disable'
                    if index == color_data_3.get():
                        color_data_3.set(data_3_global.columns.get_loc(data_chosen_columns.columns.values[0]))
                    if index == size_data_3.get():
                        size_data_3.set(data_3_global.columns.get_loc(data_chosen_columns.columns.values[0]))

                global data_3_selected_columns
                data_3_selected_columns = data_chosen_columns.copy()


def button_edit_data():
    radio_load_data1['state'] = 'disabled'
    radio_load_data2['state'] = 'disabled'
    radio_load_data3['state'] = 'disabled'
    button_edit_table['state'] = 'disabled'
    radio_PCA['state'] = 'disabled'
    radio_tsne['state'] = 'disabled'
    radio_umap['state'] = 'disabled'
    button_ok['state'] = 'normal'

    # loads the column selection
    global list_frame_data_1, list_frame_data_2
    if radio_data.get() == 1:  # -> orig_data_1 is selected
        list_frame_data_1.pack(after=frame_buttons_data)  # just shows it, creates nothing new to save memory
    elif radio_data.get() == 2:
        list_frame_data_2.pack(after=frame_buttons_data)
    elif radio_data.get() == 3:
        list_frame_data_3.pack(after=frame_buttons_data)


def button_ok_data():
    radio_load_data1['state'] = 'normal'
    radio_load_data2['state'] = 'normal'
    radio_load_data3['state'] = 'normal'
    button_edit_table['state'] = 'normal'
    radio_PCA['state'] = 'normal'
    radio_tsne['state'] = 'normal'
    radio_umap['state'] = 'normal'
    button_ok['state'] = 'disabled'

    # set that back to false before new edit round
    global selected_columns_changed
    selected_columns_changed = False

    global list_frame_data_1, list_frame_data_2
    if radio_data.get() == 1:
        list_frame_data_1.pack_forget()
    elif radio_data.get() == 2:
        list_frame_data_2.pack_forget()
    elif radio_data.get() == 3:
        list_frame_data_3.pack_forget()

    chooseSettingsForCalculation()


def changeColorScale(color_string):
    global color_scale_default
    color_scale_default = color_string

    chooseSettingsForCalculation()


def changeColorBg(color_triple):
    global color_bg_default
    color_bg_default = color_triple

    chooseSettingsForCalculation()


# when column for color or size has changed
# different from the rest, because I dont calculate new, I just redraw with different dim for color and size
# TODO: what when I dont want to have pca?!!!!!!
def quickRedraw():
    if not selected_columns_changed:  # if the selected color or size is newly added we cant show
        if radio_data.get() == 1:
            if radio_dimRed.get() == 0:
                drawCanvas(data_pca[:, 0], data_pca[:, 1],
                           data_1_selected_columns.loc[:, data_1_global.columns.values[color_data_1.get()]],
                           data_1_selected_columns.loc[:, data_1_global.columns.values[size_data_1.get()]],
                           data_1_global.columns.values[color_data_1.get()])
            elif radio_dimRed.get() == 1:
                drawCanvas(data_tsne[:, 0], data_tsne[:, 1],
                           data_1_selected_columns.loc[:, data_1_global.columns.values[color_data_1.get()]],
                           data_1_selected_columns.loc[:, data_1_global.columns.values[size_data_1.get()]],
                           data_1_global.columns.values[color_data_1.get()])
            elif radio_dimRed.get() == 2:
                drawCanvas(data_umap[:, 0], data_umap[:, 1],
                           data_1_selected_columns.loc[:, data_1_global.columns.values[color_data_1.get()]],
                           data_1_selected_columns.loc[:, data_1_global.columns.values[size_data_1.get()]],
                           data_1_global.columns.values[color_data_1.get()])
        elif radio_data.get() == 2:
            if radio_dimRed.get() == 0:
                drawCanvas(data_pca[:, 0], data_pca[:, 1],
                           data_2_selected_columns.loc[:, data_2_global.columns.values[color_data_2.get()]],
                           data_2_selected_columns.loc[:, data_2_global.columns.values[size_data_2.get()]],
                           data_2_global.columns.values[color_data_2.get()])
            elif radio_dimRed.get() == 1:
                drawCanvas(data_tsne[:, 0], data_tsne[:, 1],
                           data_2_selected_columns.loc[:, data_2_global.columns.values[color_data_2.get()]],
                           data_2_selected_columns.loc[:, data_2_global.columns.values[size_data_2.get()]],
                           data_2_global.columns.values[color_data_2.get()])
            elif radio_dimRed.get() == 2:
                drawCanvas(data_umap[:, 0], data_umap[:, 1],
                           data_2_selected_columns.loc[:, data_2_global.columns.values[color_data_2.get()]],
                           data_2_selected_columns.loc[:, data_2_global.columns.values[size_data_2.get()]],
                           data_2_global.columns.values[color_data_2.get()])
        elif radio_data.get() == 3:
            if radio_dimRed.get() == 0:
                drawCanvas(data_pca[:, 0], data_pca[:, 1],
                           data_3_selected_columns.loc[:, data_3_global.columns.values[color_data_3.get()]],
                           data_3_selected_columns.loc[:, data_3_global.columns.values[size_data_3.get()]],
                           data_3_global.columns.values[color_data_3.get()])
            elif radio_dimRed.get() == 1:
                drawCanvas(data_tsne[:, 0], data_tsne[:, 1],
                           data_3_selected_columns.loc[:, data_3_global.columns.values[color_data_3.get()]],
                           data_3_selected_columns.loc[:, data_3_global.columns.values[size_data_3.get()]],
                           data_3_global.columns.values[color_data_3.get()])
            elif radio_dimRed.get() == 2:
                drawCanvas(data_umap[:, 0], data_umap[:, 1],
                           data_3_selected_columns.loc[:, data_3_global.columns.values[color_data_3.get()]],
                           data_3_selected_columns.loc[:, data_3_global.columns.values[size_data_3.get()]],
                           data_3_global.columns.values[color_data_3.get()])
    else:
        output_text.delete('1.0', tk.END)
        output_text.insert('1.0', f'Column selection has changed since last calculation, '
                                  f'therefore the quick-view cant be shown until the new data is computed.')


# global variables / objects for the GUI ------------------------------------------------------------------------------
window = tk.Tk()

# start value is 10 -> button is not selected by default, if not clicked by user, the command functon will not activate
radio_data = tk.IntVar(value=3)  # The Dataset radio variable: 1 -> first dataset as default
radio_dimRed = tk.IntVar(value=0)  # makes pca the default selected
check_var0 = tk.IntVar(value=1)  # I said that 0 means off for this checkbox, so this enabled by default
check_var1 = tk.IntVar(value=1)
list_enable_default_1 = [True for i in range(20)]
list_enable_data_1 = [tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True)]
list_enable_default_2 = [True for i in range(16)]
list_enable_data_2 = [tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True)]
list_enable_default_3 = [True for i in range(33)]
list_enable_data_3 = [tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True),
                      tk.BooleanVar(value=True), tk.BooleanVar(value=True), tk.BooleanVar(value=True)]
color_data_1 = tk.IntVar(value=1)
size_data_1 = tk.IntVar(value=0)
color_data_2 = tk.IntVar(value=1)
size_data_2 = tk.IntVar(value=2)
color_data_3 = tk.IntVar(value=1)
size_data_3 = tk.IntVar(value=2)

menubar = tk.Menu(window)
# Perceptually Uniform Sequential: inferno, plasma, viridis -> easy to invert: viridis -> viridis_r
# Sequential: Greys, Blues, cool, copper, gist_heat, BuGn, YlGn
menu_color_scale = tk.Menu(menubar, tearoff=False, activebackground='grey')
menu_color_scale.add_radiobutton(label='Plasma', command=lambda c="plasma": changeColorScale(c), activebackground='red')
menu_color_scale.add_radiobutton(label='Viridis', command=lambda c="viridis": changeColorScale(c),
                                 activebackground='purple')
menu_color_scale.add_radiobutton(label='Inferno', command=lambda c="inferno": changeColorScale(c),
                                 activebackground='orange')
menu_color_scale.add_separator()
menu_color_scale.add_radiobutton(label='Greys', command=lambda c="Greys_r": changeColorScale(c),
                                 activebackground='grey')
menu_color_scale.add_radiobutton(label='Blues', command=lambda c="Blues_r": changeColorScale(c),
                                 activebackground='blue')
menu_color_scale.add_radiobutton(label='BuGn', command=lambda c="BuGn_r": changeColorScale(c), activebackground='green')
menubar.add_cascade(label='Color Scale', menu=menu_color_scale)
# window.configure(menu=menubar)

menu_color_bg = tk.Menu(menubar, tearoff=False, activebackground='grey')
menu_color_bg.add_radiobutton(label='White', command=lambda bg=(1.0, 1.0, 1.0): changeColorBg(bg),
                              activebackground='white')
menu_color_bg.add_radiobutton(label='Light Grey', command=lambda bg=(0.75, 0.75, 0.75): changeColorBg(bg),
                              activebackground='light grey')
menu_color_bg.add_radiobutton(label='Grey', command=lambda bg=(0.5, 0.5, 0.5): changeColorBg(bg),
                              activebackground='dark grey')
menu_color_bg.add_radiobutton(label='Dark Grey', command=lambda bg=(0.25, 0.25, 0.25): changeColorBg(bg),
                              activebackground='grey')
menu_color_bg.add_radiobutton(label='Black', command=lambda bg=(0.0, 0.0, 0.0): changeColorBg(bg),
                              activebackground='black')
menubar.add_cascade(label='Color Scale', menu=menu_color_bg)
window.configure(menu=menubar)

frame_left = ttk.Frame(window, borderwidth=10, relief=tk.RIDGE)
label_load_data = ttk.Label(frame_left, text='Load Data', font='Verdana 13 underline')
radio_load_data1 = ttk.Radiobutton(frame_left, text='Original data: 12-2018 to 05-2019', variable=radio_data, value=1, takefocus=False, command=radio_command_choseData)
radio_load_data2 = ttk.Radiobutton(frame_left, text='Original data: 12-11-2020 to 30-11-2020', variable=radio_data, value=2, takefocus=False, command=radio_command_choseData)
radio_load_data3 = ttk.Radiobutton(frame_left, text='Norm data: 12-2018 to 05-2019', variable=radio_data, value=3, takefocus=False, command=radio_command_choseData)

frame_buttons_data = ttk.Frame(frame_left)
button_edit_table = ttk.Button(frame_buttons_data, text='edit columns', takefocus=False, command=button_edit_data)
button_ok = ttk.Button(frame_buttons_data, text='ok', takefocus=False, command=button_ok_data, state='disabled')

label_dim_red = ttk.Label(frame_left, text='Dimension-Reduction', font='Verdana 13 underline')
radio_PCA = ttk.Radiobutton(frame_left, text='PCA', variable=radio_dimRed, value=0, takefocus=False,
                            command=radio_command_dimRed)
radio_tsne = ttk.Radiobutton(frame_left, text='t-SNE', variable=radio_dimRed, value=1, takefocus=False,
                             command=radio_command_dimRed)
radio_umap = ttk.Radiobutton(frame_left, text='UMAP', variable=radio_dimRed, value=2, takefocus=False,
                             command=radio_command_dimRed)
label_options = ttk.Label(frame_left, text='Options', font='Verdana 13 underline')
check_std = ttk.Checkbutton(frame_left, text='Enable Standard-Scalar for Dim-Red', variable=check_var1, onvalue=1,
                            offvalue=0, takefocus=False, command=check1, state='normal')
check_remove_nan = ttk.Checkbutton(frame_left, text='Enable removal of data with NaN values', variable=check_var0,
                                   onvalue=1, offvalue=0, takefocus=False, state='disabled')

output_text = scrolledtext.ScrolledText(window, bg='black', fg='white', font='Verdana 10')
canvas = tk.Canvas(window,
                   bg='grey')  # this bg color does not do anything -> probably has to be changed in mathplotlib...

# other global variables / objects ------------------------------------------------------------------------------------
global data_1_global  # the originally loaded data (first half of first dataset)
global data_2_global
global data_3_global
global data_1_selected_columns  # the active columns of data_1, pandas dataframe
global data_2_selected_columns
global data_3_selected_columns
global list_frame_data_1
global list_frame_data_2
global list_frame_data_3
global data_pca
global data_tsne
global data_umap
process = Process()
enable_standard_scalar = True
color_scale_default = 'plasma'
color_bg_default = (0.5, 0.5, 0.5)
selected_columns_changed = False

'''
def optimizeTSNE(data):
    temp = [x for x in range(105) if x > 2]
    for i in temp:
        data_scaled = StandardScaler().fit_transform(data)  # sets features to unit scale with mean = 0 and variance = 1
        tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=i).fit(data_scaled)
        print("Data1: perplexity =", i, " kl-divergence:", tsne.kl_divergence_)
'''


def optimizePCA():
    # assume only 3 dim -> 000, 001, 010, 011, 100, 101, 110, 111 -> 2^3 = 8 BUT 20 dim -> 2^20 = 1048576
    # if every calc takes 0.5s -> 524288 -> in hours = 145h -> it has to be cheaper ...
    # 2^12 = 4096 * 0.5s = 2048 / 3600 = 0.568h -> that would work

    # permute the dataframe -> as in: 1 for "in" and 0 for "not in" the selection of the columns
    num_bits = 12
    result = np.ones((1, num_bits), dtype=np.uint8)
    min = 3

    for i in range(2 ** num_bits):
        bits = [(i >> bit) & 1 for bit in range(num_bits - 1, -1, -1)]
        if np.sum(bits) >= min:
            result = np.append(result, np.array([bits]), axis=0)

    # deletes the first element, because we needed that only so that we actually have something to append to ...
    result = np.delete(result, 0, axis=0)


    # TODO: use 'result' to chose columns for the next optimisation



    # TODO: look up how to save to file
    # TODO: look up tabular like formatting



# My Methods ----------------------------------------------------------------------------------------------------------
# def printColumnInfo(data, index, sum_data_active_potentially_row_count):
def printColumnInfo(data, index, row_count_before_selection, row_count_after_selection):
    sum_data_active = len(data)
    sum_nan = data.iloc[:, index].isna().sum()  # amount of nan values in the selected column
    output_text.delete('1.0', tk.END)
    output_text.insert('1.0', f"Column {data.columns.values[index]} has {sum_nan} NaN values"
                              f" (of {sum_data_active} -> {round((sum_nan / sum_data_active) * 100, 2)}%)\n")

    output_text.insert('2.0', f"Under the current selection of columns, the total, non NaN, rows to be used"
                              f" for dimRed goes from {row_count_before_selection} to {row_count_after_selection} "
                              f"(100% to {round((row_count_after_selection / row_count_before_selection) * 100, 2)}%)\n")

    output_text.insert('3.0', f"The data of the column ranges from {round(data.iloc[:, index].min(), 2)} to"
                              f" {round(data.iloc[:, index].max(), 2)} with a mean of {round(data.iloc[:, index].mean(), 2)}")


def chooseSettingsForCalculation():
    if radio_data.get() == 1:
        if radio_dimRed.get() == 0:
            computePCA(data_1_selected_columns)
        elif radio_dimRed.get() == 1:
            computeTSNE(data_1_selected_columns)
        elif radio_dimRed.get() == 2:
            computeUMAP(data_1_selected_columns)
        else:
            print("Wooops, something went terrible wrong ... maybe do a restart")
    elif radio_data.get() == 2:
        if radio_dimRed.get() == 0:
            computePCA(data_2_selected_columns)
        elif radio_dimRed.get() == 1:
            computeTSNE(data_2_selected_columns)
        elif radio_dimRed.get() == 2:
            computeUMAP(data_2_selected_columns)
        else:
            print("Wooops, something went terrible wrong ... maybe do a restart")
    elif radio_data.get() == 3:
        if radio_dimRed.get() == 0:
            computePCA(data_3_selected_columns)
        elif radio_dimRed.get() == 1:
            computeTSNE(data_3_selected_columns)
        elif radio_dimRed.get() == 2:
            computeUMAP(data_3_selected_columns)
        else:
            print("Wooops, something went terrible wrong ... maybe do a restart")

    else:
        print("Wooops, something went terrible wrong ... maybe do a restart")


def drawProcessCanvas(data_x, data_y, to_map_color, to_map_size, to_map_string):
    fig, ax = plt.subplots()

    to_map_size = np.asarray(to_map_size).reshape(-1, 1)
    size = MinMaxScaler(feature_range=(1, 10), copy=True).fit_transform(np.asarray(to_map_size))

    plot = ax.scatter(np.asarray(data_x), np.asarray(data_y), c=np.asarray(to_map_color), cmap=color_scale_default,
                      s=size)
    ax.set_facecolor(color_bg_default)
    cb = fig.colorbar(plot)
    # cb.set_label(to_map_string)  # -> works
    fig.axes[1].set(xlabel=to_map_string)

    # ax.set_xlabel('x label')  # Add an x-label to the axes.
    # ax.set_ylabel('y label')  # Add a y-label to the axes.

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(row=0, column=1, rowspan=4, sticky='nesw', padx=(3, 6), pady=(6, 3))
    canvas.draw()


# draws any fig on the canvas, that is actually quite cool :)
# the extra process stuff reduces the memory leak greatly but does not to 0 ... so there is an other one???
def drawCanvas(data_x, data_y, to_map_color, to_map_size, to_map_string):
    plt.close("all")

    global process
    if process.is_alive():
        process.terminate()

    process = Process(target=drawProcessCanvas(data_x, data_y, to_map_color, to_map_size, to_map_string))
    process.start()

    # wait until proc terminates.
    # process.join()


def computeUMAP(data):
    output_text.delete('1.0', tk.END)
    output_text.insert('1.0', f'The used data has the shape: {data.shape}\n')

    data.dropna(inplace=True)

    global data_umap

    # if a random state is set -> then there wont be any parallelism ... chose speed or reproduce ability
    # n_neighbors=15 is default
    if enable_standard_scalar:
        data_scaled = StandardScaler().fit_transform(data)  # sets features to unit scale with mean = 0 and variance = 1
        data_umap = umap.UMAP(n_neighbors=20, n_components=2, random_state=42, n_jobs=1).fit_transform(data_scaled)
    else:
        data_umap = umap.UMAP(n_neighbors=20, n_components=2, random_state=42, n_jobs=1).fit_transform(data)

    if radio_data.get() == 1:
        drawCanvas(data_umap[:, 0], data_umap[:, 1],
                   data_1_selected_columns.loc[:, data_1_global.columns.values[color_data_1.get()]],
                   data_1_selected_columns.loc[:, data_1_global.columns.values[size_data_1.get()]],
                   data_1_global.columns.values[color_data_1.get()])
    elif radio_data.get() == 2:
        drawCanvas(data_umap[:, 0], data_umap[:, 1],
                   data_2_selected_columns.loc[:, data_2_global.columns.values[color_data_2.get()]],
                   data_2_selected_columns.loc[:, data_2_global.columns.values[size_data_2.get()]],
                   data_2_global.columns.values[color_data_2.get()])
    elif radio_data.get() == 3:
        drawCanvas(data_umap[:, 0], data_umap[:, 1],
                   data_3_selected_columns.loc[:, data_3_global.columns.values[color_data_3.get()]],
                   data_3_selected_columns.loc[:, data_3_global.columns.values[size_data_3.get()]],
                   data_3_global.columns.values[color_data_3.get()])


def computeTSNE(data):
    output_text.delete('1.0', tk.END)
    output_text.insert('1.0', f'The used data has the shape: {data.shape}\n')

    data.dropna(inplace=True)

    global data_tsne
    # this has to be in there bc I have to know which dataset it is for the static output
    # the kl_divergence_ has to be static, or I have to do calculate tsne 2 times ... bc y can either have the resulting
    # data or the information about the data ... I know it is really stupid -.- ... or I am, but I dont see it ...
    if radio_data.get() == 1:  # TODO: do this better ... just better ...
        if enable_standard_scalar:
            data_scaled = StandardScaler().fit_transform(data)  # sets features to unit scale with mean = 0 and variance = 1

            # init='pca' -> to preserve the global structure
            # perplexity=30 is default
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit(data_scaled)
            output_text.insert('2.0', f'Perplexity = 40 (rest is default) -> kl_divergence_ = {tsne.kl_divergence_}\n')

            data_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit_transform(data_scaled)
        else:
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit(data)
            output_text.insert('2.0', f'Perplexity = 40 (rest is default) -> kl_divergence_ = {tsne.kl_divergence_}\n')

            data_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit_transform(data)

        drawCanvas(data_tsne[:, 0], data_tsne[:, 1],
                   data_1_selected_columns.loc[:, data_1_global.columns.values[color_data_1.get()]],
                   data_1_selected_columns.loc[:, data_1_global.columns.values[size_data_1.get()]],
                   data_1_global.columns.values[color_data_1.get()])
    elif radio_data.get() == 2:
        if enable_standard_scalar:
            data_scaled = StandardScaler().fit_transform(
                data)  # sets features to unit scale with mean = 0 and variance = 1

            # init='pca' -> to preserve the global structure
            # perplexity=30 is default
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit(data_scaled)
            output_text.insert('2.0', f'Perplexity = 40 (rest is default) -> kl_divergence_ = {tsne.kl_divergence_}\n')

            data_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit_transform(data_scaled)
        else:
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit(data)
            output_text.insert('2.0', f'Perplexity = 40 (rest is default) -> kl_divergence_ = {tsne.kl_divergence_}\n')

            data_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit_transform(data)

        drawCanvas(data_tsne[:, 0], data_tsne[:, 1],
                   data_2_selected_columns.loc[:, data_2_global.columns.values[color_data_2.get()]],
                   data_2_selected_columns.loc[:, data_2_global.columns.values[size_data_2.get()]],
                   data_2_global.columns.values[color_data_2.get()])
    elif radio_data.get() == 3:
        if enable_standard_scalar:
            data_scaled = StandardScaler().fit_transform(
                data)  # sets features to unit scale with mean = 0 and variance = 1

            # init='pca' -> to preserve the global structure
            # perplexity=30 is default
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit(data_scaled)
            output_text.insert('2.0', f'Perplexity = 40 (rest is default) -> kl_divergence_ = {tsne.kl_divergence_}\n')

            data_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit_transform(data_scaled)
        else:
            tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit(data)
            output_text.insert('2.0', f'Perplexity = 40 (rest is default) -> kl_divergence_ = {tsne.kl_divergence_}\n')

            data_tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=40).fit_transform(data)

        drawCanvas(data_tsne[:, 0], data_tsne[:, 1],
                   data_3_selected_columns.loc[:, data_3_global.columns.values[color_data_3.get()]],
                   data_3_selected_columns.loc[:, data_3_global.columns.values[size_data_3.get()]],
                   data_3_global.columns.values[color_data_3.get()])


# just in principle, if you want to visualize and go down below 85% variance, it is not a accurate represantation anymore, as too much data is lost
def computePCA(data):
    # tr = tracker.SummaryTracker()

    output_text.delete('1.0', tk.END)
    pd.set_option('display.max_columns', None)

    #for series_name, series in data.items():  # print(series_name) #print(series)
        #data.dropna(subset=[series_name], inplace=True)
    data.dropna(inplace=True)

    output_text.insert('1.0', f'The used data has the shape: {data.shape}\n')

    global data_pca
    # Each principal component is a linear combination of the original variables, not the two “best” columns of data but two new columns
    pca = PCA(n_components=2)
    pca95 = PCA(.95)
    pca85 = PCA(.85)
    if enable_standard_scalar:
        data_scaled = StandardScaler().fit_transform(data)  # sets features to unit scale with mean = 0 and variance = 1
        data_pca = pca.fit_transform(data_scaled)

        pca95.fit_transform(data_scaled)
        pca85.fit_transform(data_scaled)
    else:
        data_pca = pca.fit_transform(data)

        pca95.fit_transform(data)
        pca85.fit_transform(data)

    # print('From', pca.n_features_in_, 'dimensions to', pca.n_components_, 'we keep', sum(pca.explained_variance_ratio_),'% variance, with', pca.explained_variance_ratio_, '\n')
    out1 = sum(pca.explained_variance_ratio_) * 100  # so that the % actually makes sense ...
    output_text.insert('2.0', f'From {pca.n_features_in_} to {pca.n_components_} dimensions, we keep'
                              f' {round(out1, 2)}% variance (components: {pca.explained_variance_ratio_})\n\n')

    output_text.insert('5.0', f'For 85% remaining variance, we would need {pca85.n_components_} components\n')
    output_text.insert('6.0', f'For 95% remaining variance, we would need {pca95.n_components_} components\n\n')

    # - sign does actually not mean that it is less important -> the abs() seems sensible here
    out2 = pd.DataFrame(abs(pca.components_), columns=data.columns, index=['PC-1', 'PC-2'])
    output_text.insert('8.0', f'{out2}\n')
    pd.reset_option('max_columns')

    # print(data_pca[:, size_data_1.get()]) = print(data_pca[:, 2]) -> pca are only 2 dim!

    # tr.print_diff()
    # this can NOT!!! be simplified with iloc!!!
    if radio_data.get() == 1:
        # print(color_data_1.get(), data_1_global.columns.values[color_data_1.get()])
        drawCanvas(data_pca[:, 0], data_pca[:, 1],
                   # is the same as using 'data' here, for the next 2 lines
                   data_1_selected_columns.loc[:, data_1_global.columns.values[color_data_1.get()]],
                   data_1_selected_columns.loc[:, data_1_global.columns.values[size_data_1.get()]],
                   # data.loc[:, data_1_global.columns.values[color_data_1.get()]],
                   # data.loc[:, data_1_global.columns.values[size_data_1.get()]],
                   data_1_global.columns.values[color_data_1.get()])
    elif radio_data.get() == 2:
        drawCanvas(data_pca[:, 0], data_pca[:, 1],
                   data_2_selected_columns.loc[:, data_2_global.columns.values[color_data_2.get()]],
                   data_2_selected_columns.loc[:, data_2_global.columns.values[size_data_2.get()]],
                   data_2_global.columns.values[color_data_2.get()])
    elif radio_data.get() == 3:
        drawCanvas(data_pca[:, 0], data_pca[:, 1],
                   data_3_selected_columns.loc[:, data_3_global.columns.values[color_data_3.get()]],
                   data_3_selected_columns.loc[:, data_3_global.columns.values[size_data_3.get()]],
                   data_3_global.columns.values[color_data_3.get()])
    # tr.print_diff()


# TODO: actually look for good starting values (optimize for kept variance)
def setDefaults():
    global data_1_selected_columns, data_2_selected_columns, data_3_selected_columns

    # the default selection of the columns y see, when y start the program fresh - dataset 1
    for i in range(2, 14):  # -> from 5 to 13
        list_enable_data_1[i].set(value=False)
        list_enable_default_1[i] = False
    list_enable_data_1[4].set(value=True)
    list_enable_data_1[5].set(value=True)
    list_enable_default_1[4] = True
    list_enable_default_1[5] = True
    list_enable_data_1[8].set(value=True)
    list_enable_data_1[9].set(value=True)
    list_enable_default_1[8] = True
    list_enable_default_1[9] = True

    # create the lists and then just show and hide, not creating a new one everytime
    global list_frame_data_1, color_data_1, size_data_1
    list_frame_data_1 = ListFrame(frame_left, data_1_global.columns.values, list_enable_data_1, color_data_1, size_data_1, 33)
    list_frame_data_1.pack_forget()  # bc it has the frame_left as master it is drawn, but I want to show it later

    # to splice data_1_selected_columns to the selection is tempting but strictly speaking, here we havent edited the column selection
    # buuuuuut it just makes way more sense to cut the parts of the data away that mess up the rest ... so I do it
    # and when you clicke "edit" you see what is already selected so it does not realy make sense that the alg shows all 20 dim
    # I obviosly dont want to cut from the loaded data, so I have to copy once...
    data_1_selected_columns = data_1_global.copy()
    for index, item in enumerate(list_enable_default_1):
        if not item:  # so every false boolean in list_enable_default_1 -> so 5 to 13 from 8 Lines above
            data_1_selected_columns.drop(data_1_global.columns.values[index], axis=1, inplace=True)

    # disable the color, size radio buttons of the columns that are disabled by default - for data1
    for index, item in enumerate(list_enable_default_1):
        if not item:  # so every false boolean in list_enable_default_1 -> so 5 to 13
            list_frame_data_1.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'disable'
            list_frame_data_1.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'disable'

# dataset 2 -----------------------------------------------------------------------------------------------------------

    for i in range(4, 12):  # from 4 to 11
        list_enable_data_2[i].set(value=False)
        list_enable_default_2[i] = False
    list_enable_data_2[6].set(value=True)
    list_enable_data_2[7].set(value=True)
    list_enable_default_2[6] = True
    list_enable_default_2[7] = True

    data_2_selected_columns = data_2_global.copy()
    for index, item in enumerate(list_enable_default_2):
        if not item:  # so every false boolean in list_enable_default_2
            data_2_selected_columns.drop(data_2_global.columns.values[index], axis=1, inplace=True)

    global list_frame_data_2, color_data_2, size_data_2
    list_frame_data_2 = ListFrame(frame_left, data_2_global.columns.values, list_enable_data_2, color_data_2,
                                  size_data_2, 35)
    list_frame_data_2.pack_forget()  # bc it has the frame_left as master it is drawn, but I want to show it later

    for index, item in enumerate(list_enable_default_2):
        if not item:  # so every false boolean in list_enable_default_1 -> so 5 to 13
            list_frame_data_2.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'disable'
            list_frame_data_2.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'disable'

# dataset 3 -----------------------------------------------------------------------------------------------------------

    for i in range(3, 15):
        list_enable_data_3[i].set(value=False)
        list_enable_default_3[i] = False

    data_3_selected_columns = data_3_global.copy()
    for index, item in enumerate(list_enable_default_3):
        if not item:
            data_3_selected_columns.drop(data_3_global.columns.values[index], axis=1, inplace=True)

    global list_frame_data_3, color_data_3, size_data_3
    list_frame_data_3 = ListFrame(frame_left, data_3_global.columns.values, list_enable_data_3, color_data_3, size_data_3, 32)
    list_frame_data_3.pack_forget()

    for index, item in enumerate(list_enable_default_3):
        if not item:
            list_frame_data_3.frame.winfo_children()[index + 1].winfo_children()[2]['state'] = 'disable'
            list_frame_data_3.frame.winfo_children()[index + 1].winfo_children()[3]['state'] = 'disable'


# calculate the minutes relative to the start date, which hase to be given bc of the split in the dataset
# works on the local parameter, not global variables, so that this method can be used for all data sets
# time_str is a 1-dim array with time as str obj, start_row determines the anchor point used
def calcTime(time_str, start_row):
    time_float = np.zeros(len(time_str))

    date_obj_anchor = datetime.datetime.strptime(time_str[start_row], '%Y-%m-%d %H:%M:%S')
    for id_temp, time_entry in enumerate(time_str):
        time_temp = datetime.datetime.strptime(time_entry, '%Y-%m-%d %H:%M:%S')
        time_float[id_temp] = (time_temp - date_obj_anchor).total_seconds() / 60.

    return time_float


# loads the both halfs of origData.csv in the global variable data_1_global -> 12-2018_05-2019 and data_2_global -> etc
# and cleans it up for use
def loadDatasets():
    global data_1_global, data_2_global, data_3_global, data_4_global

    data_1_global = pd.read_csv('origData.csv', sep=';', on_bad_lines='warn')
    data_3_global = pd.read_csv('normData.csv', sep=';', on_bad_lines='warn')

    # clean up data, as there are ',' for float-points ... which messes with pandas and everything else ...
    data_1_global.replace(',', '.', regex=True, inplace=True)
    data_2_global = data_1_global.copy()
    data_3_global.replace(',', '.', regex=True, inplace=True)

    # deletes all rows below 2262 -> this is because I want to split the datasets there
    data_1_global = data_1_global[data_1_global.index < 2262]
    data_2_global = data_2_global[data_2_global.index >= 2262]
    data_3_global = data_3_global[data_3_global.index < 2262]

    # dropping empty columns, 'Unnamed: 0' is an index
    data_1_global.drop(
        columns=['Unnamed: 0', '6a', '6b', '7a', '7b', '8a', '8b', '10a', '10b', '12a', '12b', '11a', '11b', '14a',
                 '14b', '13a', '13b', 'depth2'], inplace=True)
    data_2_global.drop(
        columns=['Unnamed: 0', '1a', '1b', '2a', '2b', '3a', '3b', '5a', '5b', '9a', '9b', '4a', '4b', '6a', '6b', '7a',
                 '7b', '8a', '8b', 'north', 'east', 'up'], inplace=True)
    data_3_global.drop(
        columns=['Unnamed: 0', '6a', '6b', '7a', '7b', '8a', '8b', '10a', '10b', '12a', '12b', '11a', '11b', '14a',
                 '14b', '13a', '13b', 'depth2'], inplace=True)

    # replace string time with float time (in minutes since the start) and because the halfs have very different times
    # the start should be done seperatly -> more like 2 datasets
    float_time = calcTime(data_1_global.loc[:, 'time'], 0)
    data_1_global.drop(columns=['time'], inplace=True)
    data_3_global.drop(columns=['time'], inplace=True)
    data_1_global.insert(0, 'time', float_time)
    data_3_global.insert(0, 'time', float_time)

    # the dataset is already split, so this line should be 0 ... but it is not ...
    float_time = calcTime(data_2_global.loc[:, 'time'],2262)
    data_2_global.drop(columns=['time'], inplace=True)
    data_2_global.insert(0, 'time', float_time)

    for column_name, column in data_1_global.items():
        data_1_global[column_name].replace('', np.nan, inplace=True)  # series.replace('', np.nan, inplace=True)

    for column_name, column in data_2_global.items():
        data_2_global[column_name].replace('', np.nan, inplace=True)  # series.replace('', np.nan, inplace=True)

    for column_name, column in data_3_global.items():
        data_3_global[column_name].replace('', np.nan, inplace=True)  # series.replace('', np.nan, inplace=True)

    # I dont know why I do this, but otherwise it does not work ... TODO: <- find out why ...
    data_1_global.to_csv("origData-cleaned1.csv", index=False)
    data_1_global = pd.read_csv('origData-cleaned1.csv', sep=',', on_bad_lines='warn')

    data_2_global.to_csv("origData-cleaned2.csv", index=False)
    data_2_global = pd.read_csv('origData-cleaned2.csv', sep=',', on_bad_lines='warn')

    data_3_global.to_csv("normData-cleaned3.csv", index=False)
    data_3_global = pd.read_csv('normData-cleaned3.csv', sep=',', on_bad_lines='warn')


# set up the GUI
def setupWindow():
    window.title('LoVe - Visualization tool')
    window.iconbitmap('picture.ico')

    # window size and position
    window_width = 1000
    window_height = 600
    display_width = window.winfo_screenwidth()
    # display_height = window.winfo_screenheight()
    left = int(display_width / 2 - window_width / 2)
    # top = int(display_height / 2 - window_height / 2)
    window.geometry(f'{window_width}x{window_height}+{left}+{20}')

    # Grid System -> is now a pack System bc this is easier to edit ...
    window.columnconfigure(0, weight=1, uniform='a')
    window.columnconfigure(1, weight=2, uniform='a')
    window.rowconfigure((0, 1, 2, 3, 4), weight=1, uniform='a')

    frame_left.grid(row=0, column=0, rowspan=4, sticky='nesw', padx=(6, 3), pady=(6, 3))
    label_load_data.pack(fill='x', anchor='w')
    radio_load_data1.pack(fill='x', anchor='w')
    radio_load_data3.pack(fill='x', anchor='w')
    radio_load_data2.pack(fill='x', anchor='w')

    frame_buttons_data.pack(fill='x')
    frame_buttons_data.rowconfigure(1, weight=1)
    frame_buttons_data.columnconfigure((0, 1), weight=1, uniform='a')
    button_edit_table.grid(row=0, column=0, sticky='ew')
    button_ok.grid(row=0, column=1, sticky='ew')

    label_dim_red.pack(fill='x', anchor='w')
    radio_PCA.pack(fill='x', anchor='w')
    radio_tsne.pack(fill='x', anchor='w')
    radio_umap.pack(fill='x', anchor='w')

    label_options.pack(fill='x', anchor='w')
    check_std.pack(fill='x', anchor='w')
    check_remove_nan.pack(fill='x', anchor='w')

    output_text.grid(row=4, column=0, columnspan=2, sticky='nesw', padx=(6, 6), pady=(3, 6))


if __name__ == '__main__':
    setupWindow()

    loadDatasets()

    setDefaults()

    computePCA(data_3_selected_columns)

    #optimizePCA()
    #optimizeTSNE(data_1_selected_columns)

    window.mainloop()
