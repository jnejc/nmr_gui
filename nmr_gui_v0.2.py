#######################################
# Gui program for analyzing nmr data (from 7NMR)
# created by Nejc Jansa
# nejc.jansa@ijs.si
# document creation 14.11.2016
# last version: 06.11.2018
#######################################

# Beta version (0.2)
# Memory leak not adressed, have to restart between long datasets to prevent freezing and loss of data!
# Works with analysis_v2.py

# Ideas for next version:
# Should split up code in several files for clarity
# Allow moving of files into diffent trace (for patching up bad sets)
# Marking of traces that are interesting, renaming, hiding Bad files...
# Find memory leak in tkinter/matplotlib
# Test direct execution or even transforming into .exe?
# Clear up dead buttons
# Prevent multiple plot instances from opening
# Better keyboard control
# More consistency with class naming and functions/methods
# A way of combining datasets, management of the raw data files
# Plotting specra angle dependence
# More controll over analysis parameters, make editable tables for it
# a D1 plot
# Implement parameters for smaller figure sizes to account for smaller resolution!!
# External input of new fitting formulas
# Simplify adding a sum of fits
# Remember selected fit function for each trace
# Better export format for fit results
# Select what is exported/ displayed/ plotted
# do not auto fit when opening old data (or remember the old fit function)
# add manual change of SHL and phase in T1, T2
# fix the export temperatures not matching file names (due to decimals)


# Versions of packages:
# numpy-1.11.2+mkl-cp35-cp35m-win_amd64.whl
# scipy-0.18.1-cp35-cp35m-win_amd64.whl
# matplotlib-2.0.0-cp35-cp35m-win_amd64.whl

# necessary imports
import numpy as np # for mathematic functions
import matplotlib # for plotting
import csv  # for importing and exporting csv files
import gc
### now using matplotlib 2.0.0
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # sets tkinter to be the backend of matplotlib

#matplotlib canvas and plot edit toolbar
#from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

#scipy fitting function
from scipy.optimize import curve_fit

# (Tkinter for Python 3) gui objects
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

#import analysis functions and classes
from analysis_v2 import *


#colorbrewer nice plot colot set
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999',
          '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999',
          '#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']

#set some global settings for plots
plot_font = {'family': 'Calibri', 'size': '12'}
matplotlib.rc('font', **plot_font)  # make the font settings global for all plots


### global data
#make sure things that are going to change in future and might be used in multiple places are here!
GLOBAL_experiment_dirs = ['pkl', 'raw', 'csv']
GLOBAL_t1_default_params = {'mean_range':(-4,None), 'offset':(1500,None),
                            'integral_range':(2000,2100), 'mirroring':False}
GLOBAL_t2_default_params = {'mean_range':(0,4), 'offset':(1500,None),
                            'integral_range':(2000,2100), 'mirroring':False}
GLOBAL_t1_displayed_params = ['T1', 'r', 'analysed', 'disabled', 'mirroring', 'fr', 'temp_set', 'mean_shl',
                              'mean_phase_deg', 'mean_range', 'offset_range', 'integral_range',
                              'file_key', 'file_dir', 'TAU', 'D1', 'D3', 'D9', 'NS', 'D5_min']
GLOBAL_t2_displayed_params = ['T2', 'r', 'analysed', 'disabled', 'mirroring', 'fr', 'temp_set', 'mean_shl',
                              'mean_phase_deg', 'mean_range', 'offset_range', 'integral_range',
                              'file_key', 'file_dir', 'D1', 'D3', 'D9', 'NS']
GLOBAL_spc_default_params = {'shl_start':220, 'offset':(1500,None),
                            'integral_range':(2000,2100), 'mirroring':False}
GLOBAL_spc_displayed_params = ['fr', 'broaden_width', 'fr_density', 'analysed', 'disabled', 'mirroring', 'temp_set', 'mean_shl',
                              'mean_range', 'offset_range', 'integral_range',
                              'file_key', 'file_dir', 'TAU', 'D1', 'D3', 'D9', 'NS']



class Frame_experiments(tk.Frame):
    '''Leftmost frame with selection of experiment'''
    def __init__(self, parent):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='y')

        #reference to parent
        self.parent = parent

        #load widgets
        self.Widgets()

    def Widgets(self):
        '''Puts all the widgets on the frame'''
        #adds label to frame
        self.label_experiments = tk.Label(self, text='Experiments')
        self.label_experiments.pack(side='top')
        
        #adds list to frame
        self.listbox_experiments = tk.Listbox(self, exportselection=0, bd=5,
                                              relief='flat', height=15)
        self.listbox_experiments.pack(side='top',fill='y')
        self.listbox_experiments.bind('<Return>', self.Open)
        self.listbox_experiments.bind('<F5>', self.New)
        #checks for all experiments
        for experiment in sorted(self.parent.data):
            self.listbox_experiments.insert('end', experiment)
        self.listbox_experiments.focus()
        
        #adds button to frame
        self.button_open = ttk.Button(self, text='Open', command=self.Open)
        self.button_open.pack(side='top')
        self.button_new = ttk.Button(self, text='New', command=self.New)
        self.button_new.pack(side='top')

    def New(self, event=None):
        '''Actions to perform when button_new is pressed'''

        #define button functions
        def Create(event=None):
            '''Create action, creates new experiment_data'''
            try:
                new_name = self.entry_new.get()
                path = os.path.join('data', new_name)
                os.mkdir(path)
                for sub_dir in GLOBAL_experiment_dirs:
                    os.mkdir(os.path.join(path, sub_dir))
                #adds an entry to listbox
                self.listbox_experiments.insert('end', new_name)
                #creates the Experiment_data
                self.parent.data[new_name] = Experiment_data(new_name)

                #asks for folder
                dir_new = tk.filedialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
                self.parent.data[new_name].raw_dir = dir_new
                
                self.parent.data[new_name].Add_series()
                self.parent.data[new_name].Pkl_save()
                
            except FileExistsError:
                tk.messagebox.showerror('Error','The directory already exists!')

            #forgets and removes the button and entry field
            Cancel()
            
        def Cancel(event=None):
            '''Cancel button action, removes the entry boxes'''
            self.frame_new.destroy()
            #reenables the buttons
            self.button_new.config(state='normal')
            self.button_open.config(state='normal')
            self.listbox_experiments.bind('<Return>', self.Open)
            self.listbox_experiments.bind('<F5>', self.New)
            #focus back to experiments listbox
            self.listbox_experiments.focus()

        #build the addon frame under experiments
        self.frame_new = tk.Frame(self)

        #add label
        self.label_new = tk.Label(self.frame_new, text='New experiment', bd=5)
        self.label_new.pack(side='top')

        #add entry box and set it to focus
        self.entry_new = ttk.Entry(self.frame_new, takefocus=True)
        self.entry_new.pack(side='top')
        self.entry_new.focus()
        #define enter and ecape commands within entry box
        self.entry_new.bind('<Return>', Create)
        self.entry_new.bind('<Escape>', Cancel)

        #add create button
        self.button_create = ttk.Button(self.frame_new, text='Create', command=Create)
        self.button_create.pack(side='top')
        #add cancel creation button
        self.button_cancel = ttk.Button(self.frame_new, text='Cancel', command=Cancel)
        self.button_cancel.pack(side='top')
        
        #disable the upper buttons to prevent multiple entry boxes
        self.button_new.config(state='disabled')
        self.button_open.config(state='disabled')
        self.listbox_experiments.unbind('<Return>')
        self.listbox_experiments.unbind('<F5>')

        #pack the holding frame
        self.frame_new.pack(side='top')
        

    def Open(self, event=None):
        '''Opens selected experiment and shows available series'''

        #define button functions
        def Select(event=None):
            '''Opens the traces of the selected series, loads them into listbox'''
            #get the selected series
            self.parent.current_series = self.listbox_series.get('active')
            #call traces frame functions
            self.parent.traces.Load_series()

        def Refresh(event=None):
            '''Updates raw_file_list by scanning directories again'''
            self.parent.data[self.parent.current_experiment].Find_raw_files()
            for serie in self.parent.data[self.parent.current_experiment].series:
                self.parent.data[self.parent.current_experiment].series[serie].Keys()
            msg = 'The file directory was scanned and the file lists updated!'
            tk.messagebox.showinfo('File list updated', msg)

        def Save(event=None):
            '''Save and close the current experiment'''
            self.frame_series.destroy()
            #reenables the buttons
            self.button_new.config(state='normal')
            self.button_open.config(state='normal')
            self.listbox_experiments.config(state='normal')
            self.listbox_experiments.bind('<Return>', self.Open)
            self.listbox_experiments.bind('<F5>', self.New)
            #focus back to experiments listbox
            self.listbox_experiments.focus()

            #pickle all the containing data
            self.parent.data[self.parent.current_experiment].Pkl_save()

            #close up other frames
            self.parent.traces.Disable()
            self.parent.temperatures.Disable()
            
        
        #remembers what experiment we are working on and loads it
        self.parent.current_experiment = self.listbox_experiments.get('active')
        if not self.parent.data[self.parent.current_experiment].opened:
            self.parent.data[self.parent.current_experiment].Pkl_load()
            self.parent.data[self.parent.current_experiment].opened = True


        #disable the upper buttons to prevent multiple frames
        self.button_new.config(state='disabled')
        self.button_open.config(state='disabled')
        self.listbox_experiments.config(state='disabled')
        self.listbox_experiments.unbind('<Return>')
        self.listbox_experiments.unbind('<F5>')
 
        #makes new frame for popup series
        self.frame_series = tk.Frame(self)

        #button to close and save series section
        self.button_save = ttk.Button(self.frame_series, text='Save & Close', command=Save)
        self.button_save.pack(side='top')

        #button to update file lists
        self.button_refresh = ttk.Button(self.frame_series, text='Refresh files', command=Refresh)
        self.button_refresh.pack(side='top')
        
        #add label
        self.label_series = tk.Label(self.frame_series,  text='Series', bd=5)
        self.label_series.pack(side='top')
        
        #add listbox
        self.listbox_series = tk.Listbox(self.frame_series, exportselection=0, bd=5, relief='flat')
        self.listbox_series.pack(side='top', fill='y')
        self.listbox_series.bind('<Return>', Select)
        #fill listbox
        for series in sorted(self.parent.data[self.parent.current_experiment].series):
            self.listbox_series.insert('end', series)
        self.listbox_series.focus()
        
        #add buttons
        self.button_select = ttk.Button(self.frame_series, text='Select', command=Select)
        self.button_select.pack(side='top')

        #pack popdown frame
        self.frame_series.pack(side='top')

        

class Frame_traces(tk.Frame):
    '''The frame for selecting traces and fits to display on plot'''
    def __init__(self, parent):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='y')

        #reference to parent
        self.parent = parent

        #load widgets
        self.Widgets()
        #flag to keep track if frame is already in use
        self.enabled = False

    def Widgets(self):
        '''Puts all the widgets on the frame'''
        #add label
        self.label_traces = tk.Label(self, text='Traces', state='disabled')
        self.label_traces.pack(side='top')
        
        #add listbox
        self.listbox_traces = tk.Listbox(self, exportselection=0, bd=5, height=30,
                                        relief='flat', state='disabled')
        self.listbox_traces.pack(side='top',fill='y')
        self.listbox_traces.bind('<Return>', self.Edit)

        #adds button to frame
        self.button_show = ttk.Button(self, text='Edit', command=self.Edit, state='disabled')
        self.button_show.pack(side='top')
        self.button_delete = ttk.Button(self, text='Plot', command=self.Plot,
                                        state='disabled')
        self.button_delete.pack(side='top')
        self.button_new = ttk.Button(self, text='New', command=self.New, state='disabled')
        self.button_new.pack(side='top')

    def Enable(self):
        '''Enables the items in the frame'''
        if not self.enabled:
            for child in self.winfo_children():
                child.config(state='normal')
        self.enabled = True

    def Disable(self):
        '''Disables all items in the frame'''
        if self.enabled:
            for child in self.winfo_children():
                child.config(state='disabled')
        self.enabled = False

    def Load_series(self):
        '''Actions on selection of a series'''
        self.Enable()
        self.Clear()
        for key in sorted(self.parent.data[self.parent.current_experiment].series[self.parent.current_series].keys):
            self.listbox_traces.insert('end', key)
        self.listbox_traces.focus()
        
            
    def Clear(self):
        '''Cleanup actions to do when another series is opened, or experiment is closed'''
        self.listbox_traces.delete(0, 'end')


    def Edit(self, event=None):
        '''Displayes the temperature tab of the trace and allows analysis'''
        self.parent.current_trace = self.listbox_traces.get('active')
        self.parent.temperatures.Load_trace()

    def Plot(self):
        '''Opens a frame with T1vT plot ...'''

        #disable Temperatures frame
        self.parent.temperatures.Disable()
        #disable interfering buttons
        self.button_show.config(state='disabled')
        #save selected trace
        self.parent.current_trace = self.listbox_traces.get('active')
        #T1 analysis
        if self.parent.current_series == "T1vT":
##            try:
##                if self.parent.plot_t1_t1vt.counter > 0:
##                    self.parent.plot_t1_t1vt.Add_trace(self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace])
##            except:
            self.parent.plot_t1_t1vt = Frame_plot_T1_t1vt(self.parent, self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace])
            self.parent.plot_t1_t1vt.Add_trace(self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace])        
        elif self.parent.current_series == "T2vT":
            self.parent.plot_t2_t2vt = Frame_plot_T2_t2vt(self.parent, self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace])
            self.parent.plot_t2_t2vt.Add_trace(self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace])        
        elif self.parent.current_series == "Spectrum":
            self.parent.plot_spc_frvt = Frame_plot_spc_frvt(self.parent, self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace])
            self.parent.plot_spc_frvt.Add_trace(self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace])        


    def New(self):
        Error_incomplete()
        plt.figure(figsize=(8,6))
        plt.plot([1,2,3,4,5,6,7], color=colors[1])
        plt.title("Fids")
        plt.xlabel("t (index)")
        plt.ylabel("signal")
        plt.grid()
        #print the plot
        plt.show()

    def Edit_set(self):
        '''Changes to single selection in the trace listbox and enabled edit button'''
        Error_incomplete()

    def Plot_set(self):
        '''Changes to multiple selection to allow plotting of several traces'''
        Error_incomplete()
        

class Frame_temperatures(tk.Frame):
    '''The frame for selecting traces and fits to display on plot'''
    def __init__(self, parent):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill ='y')

        #reference to parent
        self.parent = parent

        #load widgets
        self.Widgets()

        #flag to keep track if frame is already in use
        self.enabled = False
        self.wait = tk.BooleanVar(master=self, value=False)

        #memory of selected params
        self.previous_t1 = GLOBAL_t1_default_params
        self.previous_t2 = GLOBAL_t2_default_params
        self.previous_spc = GLOBAL_spc_default_params

    def Analyze_fid(self, trace):
        '''gets the (T1) trace class and runs the analysis and plotting functions'''
        #T1 analysis
        if self.parent.current_series == "T1vT":
            if trace.analysed:
                #skip to reviewing
                self.parent.plot_t1_view = Frame_plot_T1_view(self.parent, trace)
                self.wait.set(False)
            else:
                #run analysis
                self.parent.plot_t1_quick = Frame_plot_T1_quick(self.parent, trace)
        elif self.parent.current_series == "T2vT":
            if trace.analysed:
                #skip to reviewing
                self.parent.plot_t2_view = Frame_plot_T2_view(self.parent, trace)
                self.wait.set(False)
            else:
                #run analysis
                self.parent.plot_t2_quick = Frame_plot_T2_quick(self.parent, trace)
        elif self.parent.current_series == "Spectrum":
            if trace.analysed:
                #skip to reviewing
                self.parent.plot_spc_view = Frame_plot_spc_view(self.parent, trace)
                self.wait.set(False)
            else:
                #run analysis
                self.parent.plot_spc_quick = Frame_plot_spc_quick(self.parent, trace)
        else:
            Error_incomplete()
            self.parent.temperatures.wait.set(False)
            self.button_show.config(state='normal')
            self.parent.traces.button_show.config(state='normal')
            #refresh the temperatures tab
            self.parent.temperatures.Load_trace()
            
    def Widgets(self):
        '''Puts all the widgets on the frame'''
        #button functions
        def Show(action=None):
            '''Opens analysis window for selected temperatures'''
            #disable buttons that could interrupt loop
            self.button_show.config(state='disabled')
            self.parent.traces.button_show.config(state='disabled')
            #run analysis loop
            for select in self.listbox_temperatures.curselection():
                temp = self.listbox_temperatures.get(select)
                self.wait.set(True)
                self.Analyze_fid(self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace][temp])
                #wait untill the analysis is finished before continuing the loop!
                root.wait_variable(self.wait)
            #reenable buttons
            self.button_show.config(state='normal')
            self.parent.traces.button_show.config(state='normal')
            #refresh the temperatures tab
            self.parent.temperatures.Load_trace()
       
        def Delete():
            '''Deletes the reference to the selected temperatures'''
            #loop over selected files
            for select in self.listbox_temperatures.curselection():
                temp = self.listbox_temperatures.get(select)
                self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace].pop(temp, None)

            #refresh list
            self.Load_trace()

        def Deselect(action=None):
            '''Deselects the active entries in listbox'''
            self.listbox_temperatures.selection_clear(0,'end')
    
        #add label
        self.label_temperatures = tk.Label(self,  text='Temperatures', state='disabled')
        self.label_temperatures.pack(side='top')

        #listbox frame
        self.frame_listbox = tk.Frame(self)
        self.frame_listbox.pack(side='top', fill='y')
        #add listbox
        self.listbox_temperatures = tk.Listbox(self.frame_listbox, selectmode='extended', exportselection=0,
                                               bd=5, relief='flat', state='disabled', height=30)
        self.listbox_temperatures.pack(side='left',fill='y')
        #keybinds for listbox
        self.listbox_temperatures.bind('<Return>', Show)
        self.listbox_temperatures.bind('<Escape>', Deselect)

        #add scrollbar
        self.scrollbar_listbox = ttk.Scrollbar(self.frame_listbox, orient='vertical')
        self.scrollbar_listbox.config(command=self.listbox_temperatures.yview)
        self.scrollbar_listbox.pack(side='right',fill='y')
        self.listbox_temperatures.config(yscrollcommand=self.scrollbar_listbox.set)

        #adds button to frame
        self.button_show = ttk.Button(self, text='Show', command=Show, state='disabled')
        self.button_show.pack(side='top')
        self.button_deselect = ttk.Button(self, text='Deselect', command=Deselect,
                                        state='disabled')
        self.button_deselect.pack(side='top')
        self.button_delete = ttk.Button(self, text='Delete', command=Delete, state='disabled')
        self.button_delete.pack(side='top')

    def Enable(self):
        '''Enables the items in the frame'''
        if not self.enabled:
            for child in self.winfo_children():
                try:
                    child.config(state='normal')
                except: pass
            self.listbox_temperatures.config(state='normal')
        self.enabled = True

    def Disable(self):
        '''disables the items in the frame'''
        if self.enabled:
            for child in self.winfo_children():
                try:
                    child.config(state='disabled')
                except: pass
            self.listbox_temperatures.config(state='disabled')
        self.enabled = False
        
    def Load_trace(self):
        '''Actions on selection of editing a trace'''
        if self.enabled:
            self.Clear()
        elif not self.enabled:
            self.Enable()
        self.Clear()
        for temp in sorted(self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace]):
            self.listbox_temperatures.insert('end', temp)
        self.listbox_temperatures.focus()

        for i, temp in enumerate(self.listbox_temperatures.get(0, 'end')):
            if self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace][temp].analysed:
                self.listbox_temperatures.itemconfig(i, bg='pale green',
                                                     selectbackground='dark green')
            #try:
            if self.parent.data[self.parent.current_experiment].series[self.parent.current_series].traces[self.parent.current_trace][temp].disabled:
                self.listbox_temperatures.itemconfig(i, bg='light salmon',
                                                     selectbackground='red')
            #except: pass

    def Clear(self):
        '''Cleanup actions to do when another temp is opened, or experiment is closed'''
        self.listbox_temperatures.delete(0, 'end')

    
class Frame_plot_T1_quick(tk.Frame):
    '''Pioneer first T1 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='both', expand=True)
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace

        #starting data
        self.range = self.parent.temperatures.previous_t1['mean_range'][0]

        #load widgets
        self.Widgets()

        #run quick t1
        quick_tables = trace.Quick_T1()
        self.Fill_plots(*quick_tables)

        #take focus away from listbox
        self.focus()
        #global key binds
        root.bind('<Left>', self.Interrupt)
        root.bind('<Right>', self.Finish)
        


    def Finish(self, event=None):
        '''Accepts the data on this screen and closes it up'''
        #save data
        self.trace.mean_range = (self.range, None)
        self.trace.mean_shl = int(self.mean_shl)
        self.trace.mean_phase = self.mean_phase
        self.parent.temperatures.previous_t1['mean_range']=(min(self.range,19), None)
        #hide this frame
        self.pack_forget()
        #close plots
        plt.close('all')
        #forget global key bind
        root.unbind('<Right>')
        root.unbind('<Left>')

        #run next frame
        self.parent.plot_t1_ranges = Frame_plot_T1_ranges(self.parent, self.trace)

    def Interrupt(self, event=None):
        '''Stops the analysis loop'''
        #Destroy frame and plots
        self.pack_forget()
        self.destroy()
        plt.close('all')
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')

        #stop the analysis loop
        self.parent.temperatures.wait.set(False)

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #split in two half frames
        self.frame_left = tk.Frame(self)
        self.frame_right = tk.Frame(self)
        self.frame_left.pack(side='left', fill='y')
        self.frame_right.pack(side='left', fill='y')

        #add frames on left side
        self.frame_left1 = tk.Frame(self.frame_left, bd=5)
        self.frame_left2 = tk.Frame(self.frame_left, bd=5)
        self.frame_left3 = tk.Frame(self.frame_left, bd=5)
        self.frame_left1.pack(side='top')
        self.frame_left2.pack(side='top')
        self.frame_left3.pack(side='top', fill='x')
        #add frames on right side
        self.frame_right1 = tk.Frame(self.frame_right, bd=5)
        self.frame_right2 = tk.Frame(self.frame_right, bd=5)
        self.frame_right1.pack(side='top')
        self.frame_right2.pack(side='top')

        #add canvases and toolbars
        #plot 1
        self.fig_left1 = plt.figure(dpi=100, figsize=(7,3))
        self.fig_left1.subplots_adjust(bottom=0.20, left= 0.14, right=0.96, top=0.88)
        self.fig_left1.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')
        self.canvas_left1 = FigureCanvasTkAgg(self.fig_left1, self.frame_left1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left1, self.frame_left1)
        self.canvas_left1._tkcanvas.pack()

        #plot 2
        self.fig_left2 = plt.figure(dpi=100, figsize=(7,4))
        self.fig_left2.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.9)
        self.canvas_left2 = FigureCanvasTkAgg(self.fig_left2, self.frame_left2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left2, self.frame_left2)
        self.canvas_left2._tkcanvas.pack()
        
        #interrupt button
        self.button_interrupt = ttk.Button(self.frame_left3, text='Interrupt', command=self.Interrupt)
        self.button_interrupt.pack(side='left', anchor='w')

        #label and edit of mean_range
        self.frame_left3_middle = tk.Frame(self.frame_left3)
        self.frame_left3_middle.pack(anchor='center')

        self.label_mean = tk.Label(self.frame_left3_middle,  text='Selected range:')
        self.label_mean.pack(side='left')

        self.entry_mean_var = tk.StringVar(self, value=self.range)
        self.entry_mean = ttk.Entry(self.frame_left3_middle,
                                    textvariable=self.entry_mean_var, width=3)
        self.entry_mean.pack(side='left')


        #plot 3
        self.fig_right1 = plt.figure(dpi=100, figsize=(7,3.5))
        self.fig_right1.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.9)
        self.canvas_right1 = FigureCanvasTkAgg(self.fig_right1, self.frame_right1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right1, self.frame_right1)
        self.canvas_right1._tkcanvas.pack()

        #plot 4
        self.fig_right2 = plt.figure(dpi=100, figsize=(7,3.5))
        self.fig_right2.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.9)
        self.canvas_right2 = FigureCanvasTkAgg(self.fig_right2, self.frame_right2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right2, self.frame_right2)
        self.canvas_right2._tkcanvas.pack()


        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_right, text='Confirm', command=self.Finish)
        self.button_confirm.pack(side='top', anchor='ne')

    def Fill_plots(self, temp_list, temp_list2, temp_set, tau_list, t1_list, phase_list, shl_list):
        '''Puts the contents into the plot fields'''
        #starting values
        self.mean_t1 = np.mean(t1_list[self.range:])
        self.mean_phase = np.mean(np.unwrap(phase_list[self.range:]))
        self.mean_shl = np.round(np.mean(shl_list[self.range:]))
        #x axes
        n = len(tau_list)
        x_list = np.linspace(1,n,n)
        
        #plot 1, temperature stabillity    
        self.axes_left1 = self.fig_left1.add_subplot(111)
        if abs(np.mean(temp_list) - temp_set) < 2:
            self.axes_left1.plot(x_list, temp_list, marker='.', color=colors[1], label='ITC_R1')
        if abs(np.mean(temp_list2) - temp_set) < 2:
            self.axes_left1.plot(x_list, temp_list2, marker='.', color=colors[2], label='ITC_R2')
        self.axes_left1.axhline(y=temp_set, color=colors[0], label='Set T')
        self.axes_left1.margins(0.02, 0.1)
        self.axes_left1.set_title('Temperature stabillity check')
        self.axes_left1.set_xlabel('File index')
        self.axes_left1.set_ylabel('Temperature (K)')
        self.axes_left1.legend(loc='upper right')
        self.axes_left1.grid()

        #plot 2 quick T1 points
        self.axes_left2 = self.fig_left2.add_subplot(111)
        self.axes_left2.plot(tau_list, t1_list, 'bo', color=colors[1], label='Data')
        self.axes_left2_vline = self.axes_left2.axvline(x=tau_list[self.range],
                                                        color=colors[2], label='Select')
        self.axes_left2_hline = self.axes_left2.axhline(y=self.mean_t1, color=colors[0],
                                                        label='Plato')
        self.axes_left2.set_xscale('log')
        self.axes_left2.set_title('T1 quick check')
        self.axes_left2.set_xlabel(r'$\tau$ ($\mu$s)')
        self.axes_left2.set_ylabel('Signal')
        #legend = self.axes_left2.legend(loc='lower right')
        #legend.draggable()
        self.axes_left2.grid()
        
        #plot 3 quick phases
        self.axes_right1 = self.fig_right1.add_subplot(111)
        self.axes_right1.plot(x_list, np.unwrap(np.array(phase_list))*180/np.pi, marker='.',
                             color=colors[1], label='Phase')
        self.axes_right1_hline = self.axes_right1.axhline(self.mean_phase*180/np.pi, color=colors[0], label='Mean phase')
        self.axes_right1.margins(0.02, 0.1)
        self.axes_right1.set_title('Phase check')
        self.axes_right1.set_xlabel('File index')
        self.axes_right1.set_ylabel('Phase (Deg)')
        #self.axes_right1.legend(loc='lower right')
        self.axes_right1.grid()

        #plot 4 quick shl
        self.axes_right2 = self.fig_right2.add_subplot(111)
        self.axes_right2.plot(x_list, shl_list, marker='.',
                             color=colors[1], label='SHL')
        self.axes_right2_hline = self.axes_right2.axhline(self.mean_shl,
                                                          color=colors[0], label='Mean SHL')
        self.axes_right2.margins(0.02, 0.1)
        self.axes_right2.set_title('SHL check')
        self.axes_right2.set_xlabel('File index')
        self.axes_right2.set_ylabel('Shift left')
        #self.axes_right2.legend(loc='lower right')
        self.axes_right2.grid()

        #redraw canvases
        self.fig_left1.canvas.draw()
        self.fig_left2.canvas.draw()
        self.fig_right1.canvas.draw()
        self.fig_right2.canvas.draw()
        

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, recalculates mean of selected points'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.range = np.searchsorted(tau_list, event.xdata, side='right')
                self.mean_t1 = np.mean(t1_list[self.range:])
                self.mean_phase = np.mean(np.unwrap(phase_list[self.range:]))
                self.mean_shl = np.round(np.mean(shl_list[self.range:]))
                self.entry_mean_var.set(self.range)
                #update plot
                self.axes_left2_vline.set_xdata(event.xdata)
                self.axes_left2_hline.set_ydata(self.mean_t1)
                self.axes_right1_hline.set_ydata(self.mean_phase*180/np.pi)
                self.axes_right2_hline.set_ydata(self.mean_shl)
                self.fig_left2.canvas.draw()
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()

        self.axes_left2_vline_drag = self.fig_left2.canvas.mpl_connect('motion_notify_event', Drag)

class Frame_plot_T1_ranges(tk.Frame):
    '''Pioneer first T1 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='both', expand=True)
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace

        self.offset_select = self.parent.temperatures.previous_t1['offset'][0]
        self.range_l_select = self.parent.temperatures.previous_t1['integral_range'][0]
        self.range_r_select = self.parent.temperatures.previous_t1['integral_range'][1]
        self.mirroring = self.parent.temperatures.previous_t1['mirroring']

        self.shl_select = self.trace.mean_shl
        
        #load widgets
        self.Widgets()

        #load plots and read
        self.Choose_offset(trace)

        self.focus()
        #global key bindings
        root.bind('<Left>', self.Previous)
        root.bind('<Right>', self.Confirm_offset)
        

    def Widgets(self):
        '''Builds all the subframes and canvases'''
     
        def Set_offset(event=None):
            '''Entry change of offset, replot and write value'''
            try:
                self.offset_select = int(self.entry_offset.get())
                #update plot
                self.axes_left1_vline.set_xdata(self.offset_select)
                self.axes_left2_vline.set_xdata(self.offset_select)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted values must be integers!')

        def Set_range(event=None):
            '''Entry change of ranges, replot and save value'''
            try:
                self.range_l_select = int(self.entry_range_l_var.get())
                self.range_r_select = int(self.entry_range_r_var.get())
                self.axes_right1_vline_l.set_xdata(self.spc_fr[self.range_l_select])
                self.axes_right2_vline_l.set_xdata(self.spc_fr[self.range_l_select])
                self.axes_right1_vline_r.set_xdata(self.spc_fr[self.range_r_select])
                self.axes_right2_vline_r.set_xdata(self.spc_fr[self.range_r_select])
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted values must be integers!')

        def Set_shl(event=None):
            '''Entry change of shl, replot and write value'''
            try:
                self.shl = int(self.entry_shl.get())
                #update plot
                self.axes_left1_vline_shl.set_xdata(self.shl)
                self.axes_left2_vline_shl.set_xdata(self.shl)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted values must be integers!')
                

        #split in two half frames
        self.frame_left = tk.Frame(self)
        self.frame_right = tk.Frame(self)
        self.frame_left.pack(side='left', fill='y')
        self.frame_right.pack(side='left', fill='y')

        #add frames on left side
        self.frame_left1 = tk.Frame(self.frame_left, bd=5)
        self.frame_left2 = tk.Frame(self.frame_left, bd=5)
        self.frame_left3 = tk.Frame(self.frame_left, bd=5)
        self.frame_left1.pack(side='top')
        self.frame_left2.pack(side='top')
        self.frame_left3.pack(side='top', fill='x')
        #add frames on right side
        self.frame_right1 = tk.Frame(self.frame_right, bd=5)
        self.frame_right2 = tk.Frame(self.frame_right, bd=5)
        self.frame_right3 = tk.Frame(self.frame_right, bd=5)
        self.frame_right1.pack(side='top')
        self.frame_right2.pack(side='top')
        self.frame_right3.pack(side='top', fill='x')

        #add canvases and toolbars
        #plot 1
        self.fig_left1 = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_left1.subplots_adjust(bottom=0.20, left= 0.10, right=0.96, top=0.88)
        self.fig_left1.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')
        self.canvas_left1 = FigureCanvasTkAgg(self.fig_left1, self.frame_left1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left1, self.frame_left1)
        self.canvas_left1._tkcanvas.pack()

        #plot 2
        self.fig_left2 = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_left2.subplots_adjust(bottom=0.12, left= 0.10, right=0.96, top=0.93)
        self.canvas_left2 = FigureCanvasTkAgg(self.fig_left2, self.frame_left2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left2, self.frame_left2)
        self.canvas_left2._tkcanvas.pack()
 
        #buttons left
        self.button_previous = ttk.Button(self.frame_left3, text='Repeat previous', command=self.Previous)
        self.button_previous.pack(side='left')

        self.button_confirm = ttk.Button(self.frame_left3, text='Confirm', command=self.Confirm_offset)
        self.button_confirm.pack(side='right')

        #check button for mirroring fid
        self.check_mirroring_var = tk.BooleanVar(self, False)
        if self.mirroring:
            self.check_mirroring_var.set(True)
            
        self.check_mirroring = (ttk.Checkbutton(self.frame_left3, variable=self.check_mirroring_var))
        self.check_mirroring.pack(side='right')
        
        self.label_mirroring = tk.Label(self.frame_left3,  text='Mirroring')
        self.label_mirroring.pack(side='right')
        
        #middle frame
        self.frame_left3_middle = tk.Frame(self.frame_left3)
        self.frame_left3_middle.pack(anchor='center')

        self.label_offset = tk.Label(self.frame_left3_middle,  text='Selected offset:')
        self.label_offset.pack(side='left')
        
        self.entry_offset_var = tk.StringVar(self, value=self.offset_select)
        self.entry_offset = ttk.Entry(self.frame_left3_middle,
                                      textvariable=self.entry_offset_var, width=5)
        self.entry_offset.pack(side='left')
        self.entry_offset.bind('<Return>', Set_offset)

        self.button_set_offset = ttk.Button(self.frame_left3_middle,
                                            text='Set offset', command=Set_offset)
        self.button_set_offset.pack(side='left')

        self.label_shl = tk.Label(self.frame_left3_middle,  text='Selected shl:')
        self.label_shl.pack(side='left')
        
        self.entry_shl_var = tk.StringVar(self, value=self.shl_select)
        self.entry_shl = ttk.Entry(self.frame_left3_middle,
                                      textvariable=self.entry_shl_var, width=5)
        self.entry_shl.pack(side='left')
        self.entry_shl.bind('<Return>', Set_shl)

        self.button_set_shl = ttk.Button(self.frame_left3_middle,
                                            text='Set shl', command=Set_shl)
        self.button_set_shl.pack(side='left')


        #plot 3
        self.fig_right1 = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_right1.subplots_adjust(bottom=0.20, left= 0.10, right=0.96, top=0.88)
        self.canvas_right1 = FigureCanvasTkAgg(self.fig_right1, self.frame_right1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right1, self.frame_right1)
        self.canvas_right1._tkcanvas.pack()

        #plot 4
        self.fig_right2 = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_right2.subplots_adjust(bottom=0.12, left= 0.10, right=0.96, top=0.93)
        self.canvas_right2 = FigureCanvasTkAgg(self.fig_right2, self.frame_right2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right2, self.frame_right2)
        self.canvas_right2._tkcanvas.pack()

        #buttons right
        self.label_range = tk.Label(self.frame_right3,  text='Selected ranges:')
        self.label_range.pack(side='left')
        
        self.entry_range_l_var = tk.StringVar(self, value=self.range_l_select)
        self.entry_range_l = ttk.Entry(self.frame_right3,
                                       textvariable=self.entry_range_l_var, width=5)
        self.entry_range_l.pack(side='left')
        self.entry_range_l.bind('<Return>', Set_range)

        self.label_range_comma = tk.Label(self.frame_right3,  text=' , ')
        self.label_range_comma.pack(side='left')

        self.entry_range_r_var = tk.StringVar(self, value=self.range_r_select)
        self.entry_range_r = ttk.Entry(self.frame_right3,
                                       textvariable=self.entry_range_r_var, width=5)
        self.entry_range_r.pack(side='left')
        self.entry_range_r.bind('<Return>', Set_range)

        self.button_set_range = ttk.Button(self.frame_right3, text='Set range', command=Set_range)
        self.button_set_range.pack(side='left')

        self.button_close = ttk.Button(self.frame_right3, text='Confirm',
                                       command=self.Close, state='disabled')
        self.button_close.pack(side='right')

    #button commands
    def Previous(self, event=None):
        '''Back to the previous step!'''
        #reload offset
        self.parent.plot_t1_quick.pack(side='left', fill='both', expand=True)
        #destroy me
        self.pack_forget()
        self.destroy()
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')


    def Confirm_offset(self, event=None):
        '''Saves current offset range, shl and opens integral ranges select'''
        self.trace.offset_range = (self.offset_select, None)
        self.trace.mean_shl = self.shl_select
        self.parent.temperatures.previous_t1['offset'] = (self.offset_select, None)

        #remember mirroring
        self.parent.temperatures.previous_t1['mirroring'] = self.check_mirroring_var.get()
        self.trace.mirroring = self.check_mirroring_var.get()

        #run integral ranges select and clean up buttons
        self.Choose_ranges(self.trace)
        self.button_confirm.config(state='disabled')
        self.button_close.config(state='enabled')
        self.button_close.focus_set()

        #change global keys
        root.bind('<Right>', self.Close)

    def Close(self, event=None):
        '''Confirm the selection in this screen'''
        #save the integral ranges
        self.trace.integral_range = (self.range_l_select, self.range_r_select)
        self.parent.temperatures.previous_t1['integral_range'] = (self.range_l_select,
                                                               self.range_r_select)

        #finish the analysis
        self.trace.Run()
        
        #unpack and destroy
        self.trace.analysed = True
        self.parent.plot_t1_quick.destroy()
        self.pack_forget()
        self.destroy()
        plt.close('all')
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')

        #load the overview frame
        self.parent.plot_t1_view = Frame_plot_T1_view(self.parent, self.trace)
        #self.parent.plot_t1_view.pack(side='left', fill='both', expand=True)

    def Choose_offset(self, trace):
        '''Operations and plotting for choosing the FID offsets'''
        fids = list()
        for file in trace.file_list:
            fid = FID(file, trace.file_dir)
            fids.append(fid.x)

        x_mean = np.mean(fids[slice(*trace.mean_range)], axis=0)

        #plot 1
        self.axes_left1 = self.fig_left1.add_subplot(111)
        self.axes_left1.plot(np.real(x_mean), color=colors[1], label='Re')
        self.axes_left1.plot(np.imag(x_mean), color=colors[2], label='Im')
        self.axes_left1.plot(np.abs(x_mean), color=colors[0], label='Abs')
        self.axes_left1_vline_shl = self.axes_left1.axvline(x=trace.mean_shl, color=colors[-1])
        self.axes_left1_vline = self.axes_left1.axvline(x=self.offset_select, color=colors[4])
        self.axes_left1.margins(0.02, 0.1)
        self.axes_left1.set_title('Mean FID')
        self.axes_left1.set_xlabel('Time (index)')
        self.axes_left1.set_ylabel('Signal (A.U.)')
        #self.axes_left1.legend(loc='upper right')
        self.axes_left1.grid()

        #plot 2
        self.axes_left2 = self.fig_left2.add_subplot(111, sharex=self.axes_left1)
        for i, fid in enumerate(fids):
            self.axes_left2.plot(np.abs(fid)+np.amax(np.abs(x_mean))*0.5*i,
                                 color=colors[i%9], label=str(i))
        self.axes_left2_vline_shl = self.axes_left2.axvline(x=trace.mean_shl, color=colors[-1], label='shl')
        self.axes_left2_vline = self.axes_left2.axvline(x=self.offset_select,
                                                        color=colors[4], label='Select')
        self.axes_left2.margins(0.02, 0.02)
        self.axes_left2.set_title('All FIDs')
        self.axes_left2.set_xlabel('Time (index)')
        self.axes_left2.set_ylabel('Absolute signal (A.U.)')
        self.axes_left2.grid()

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, recalculates mean of selected points'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.shl_select = int(event.xdata)
                self.entry_shl_var.set(self.shl_select)
                #update plot
                self.axes_left1_vline_shl.set_xdata(event.xdata)
                self.axes_left2_vline_shl.set_xdata(event.xdata)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()

            if event.button == 3 and event.inaxes != None:
                #find the index of selected points
                self.offset_select = int(event.xdata)
                self.entry_offset_var.set(self.offset_select)
                #update plot
                self.axes_left1_vline.set_xdata(event.xdata)
                self.axes_left2_vline.set_xdata(event.xdata)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()

        self.axes_left1_vline_drag = self.fig_left1.canvas.mpl_connect('motion_notify_event', Drag)
        self.axes_left2_vline_drag = self.fig_left2.canvas.mpl_connect('motion_notify_event', Drag)

    def Choose_ranges(self, trace):
        '''Operations and plotting for choosing spectrum integral ranges'''
        spcs = list()
        for file in trace.file_list:
            fid = FID(file, trace.file_dir)
            fid.Offset(trace.offset_range)
            fid.Shift_left(trace.mean_shl, mirroring=trace.mirroring)
            fid.Fourier()
            fid.Phase_rotate(trace.mean_phase)

            spcs.append(fid.spc)

        spc_fr = fid.spc_fr
        self.spc_fr = spc_fr
        spc_mean = np.mean(spcs[slice(*trace.mean_range)], axis=0)

        #plot 3
        self.axes_right1 = self.fig_right1.add_subplot(111)
        self.axes_right1.plot(spc_fr, np.real(spc_mean), color=colors[1], label='Re')
        self.axes_right1.plot(spc_fr, np.imag(spc_mean), color=colors[2], label='Im')
        self.axes_right1.axvline(x=trace.fr, color=colors[-1])
        self.axes_right1_vline_l = self.axes_right1.axvline(x=spc_fr[self.range_l_select],
                                                            color=colors[4])
        self.axes_right1_vline_r = self.axes_right1.axvline(x=spc_fr[self.range_r_select],
                                                            color=colors[4])
        self.axes_right1.set_xlim((trace.fr -0.5,+ trace.fr +0.5))
        self.axes_right1.set_title('Mean spectrum (Drag with left and right mouse button)')
        self.axes_right1.set_xlabel('Frequency (MHz)')
        self.axes_right1.set_ylabel('Signal (A.U.)')
        self.axes_right1.legend(loc='upper left')
        self.axes_right1.grid()

        #plot 4
        self.axes_right2 = self.fig_right2.add_subplot(111)
        for i, spc in enumerate(spcs):
            self.axes_right2.plot(spc_fr, np.real(spc)+np.amax(np.abs(spc_mean))*0.5*i,
                                  color=colors[i%9], label=str(i))
        self.axes_right1.axvline(x=trace.fr, color=colors[-1])
        self.axes_right2_vline_l = self.axes_right2.axvline(x=spc_fr[self.range_l_select],
                                                            color=colors[4])
        self.axes_right2_vline_r = self.axes_right2.axvline(x=spc_fr[self.range_r_select],
                                                            color=colors[4])
        self.axes_right2.set_xlim((trace.fr -0.5,+ trace.fr +0.5))
        self.axes_right2.margins(0.02, 0.02)
        self.axes_right2.set_title('All FIDs')
        self.axes_right2.set_xlabel('Frequency (MHz)')
        self.axes_right2.set_ylabel('Real part of signal (A.U.)')
        self.axes_right2.grid()

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, recalculates mean of selected points'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.range_l_select = np.searchsorted(spc_fr, event.xdata, side='left')
                self.entry_range_l_var.set(self.range_l_select)
                #update plot
                self.axes_right1_vline_l.set_xdata(event.xdata)
                self.axes_right2_vline_l.set_xdata(event.xdata)
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()

            if event.button == 3 and event.inaxes != None:
                #find the index of selected points
                self.range_r_select = np.searchsorted(spc_fr, event.xdata, side='right')
                self.entry_range_r_var.set(self.range_r_select)
                #update plot
                self.axes_right1_vline_r.set_xdata(event.xdata)
                self.axes_right2_vline_r.set_xdata(event.xdata)
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()

        self.axes_right1_vline_drag = self.fig_right1.canvas.mpl_connect('motion_notify_event', Drag)
        self.axes_right2_vline_drag = self.fig_right2.canvas.mpl_connect('motion_notify_event', Drag)

        self.fig_right1.canvas.draw()
        self.fig_right2.canvas.draw()
        


class Frame_plot_T1_view(tk.Frame):
    '''Pioneer first T1 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent)
        self.pack(side='left', anchor='n')
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace
        
        #load widgets
        self.Widgets()

        #global key bind
        root.bind('<Right>', self.Confirm)

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #button commands

        def Disable(event=None):
            '''Disables and red-flags the point to avoid plotting'''
            #try:
            self.trace.disabled = not self.trace.disabled
            #except:
            #    self.trace.disabled = True
            self.Refresh_parameters()

        def Repeat(event=None):
            '''Clears the T1 trace and starts the analysis from scratch'''
            self.trace.Reinit()
            self.Confirm()
            self.trace.analysed = False

            
        #bottom button row
        self.frame_bottom = tk.Frame(self)
        self.frame_bottom.pack(side='bottom', fill='x')
        #split in columns
        self.frame_parameters = tk.Frame(self, bd=5)
        self.frame_parameters.pack(side='left', anchor='n')
        self.frame_plot = tk.Frame(self, bd=5)
        self.frame_plot.pack(side='left', anchor='n')


        #parameters
        self.label_parameters = tk.Label(self.frame_parameters,  text='Parameters')
        self.label_parameters.pack(side='top')

        self.tree_columns = ('Name','Value')
        self.tree_parameters = ttk.Treeview(self.frame_parameters, columns=self.tree_columns,
                                            show='headings', selectmode='none', height=25)
        self.tree_parameters.pack(side='top',fill='y', expand=True)

        #define column widths
        self.tree_parameters.column('Name', width=80)
        self.tree_parameters.column('Value', width=120)
        #define column names
        for column in self.tree_columns:
            self.tree_parameters.heading(column, text=column)
        #display in degrees
        self.trace.mean_phase_deg = self.trace.mean_phase*180/np.pi
        #fill in params
        self.Refresh_parameters()
        
        # disable point button
        self.button_disable = ttk.Button(self.frame_parameters, text='Disable/enable Point',
                                         command=Disable, width=20)
        self.button_disable.pack(side='top')
        #redo analysis button
        self.button_repeat = ttk.Button(self.frame_parameters, text='Repeat analysis',
                                        command=Repeat, width=20)
        self.button_repeat.pack(side='top')

        #T1 plot
        self.fig_t1 = plt.figure(dpi=100, figsize=(8,6))
        self.fig_t1.subplots_adjust(bottom=0.1, left= 0.10, right=0.96, top=0.94)
        self.fig_t1.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')
        #self.fig_t1.text(0.82, 0.97, r'$y_0(1-(1-s) \exp(-(\frac{x}{T_1})^r))$', horizontalalignment='center', verticalalignment='center')


        self.canvas_t1 = FigureCanvasTkAgg(self.fig_t1, self.frame_plot)
        self.canvas_t1._tkcanvas.pack()

        self.Fill_plot()
        self.Fitting_frame()
        

    def Fitting_frame(self, event=None):
        '''Repacks/initializes the fitting frame for the selected fitting function'''

        #repack if existing:
        try:
            self.frame_fit.destroy()
        except:
            pass
        #fit frame
        self.frame_fit = tk.Frame(self, bd=5)
        self.frame_fit.pack(side='left', anchor='n', fill='y')

        #fit functions
        def Fit_exponential(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 exponential fit model'''
            return y0*(1-(1+s)*np.exp(-(x/T1)**r))

        def Fit_spin_3_2(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 fit model for spin 3/2'''
            return y0*(1-(1+s)*(0.1*np.exp(-(x/T1)**r)+0.9*np.exp(-(6*x/T1)**r)))

        def Fit_spin_3_2_dbl(x, T11=0.001, T12=0.001, y0=1000, s1=1, s2=1, r=1):
            '''T1 fit model for spin 3/2'''
            return y0*(1-(1+s1)*(0.1*np.exp(-(x/T11)**r)+0.9*np.exp(-(6*x/T11)**r)) \
                   -(1+s2)*(0.1*np.exp(-(x/T12)**r)+0.9*np.exp(-(6*x/T12)**r)))

        def Fit_spin_3_2_1st(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 fit model for spin 3/2'''
            return y0*(1-(1+s)*(0.1*np.exp(-(x/T1)**r)+0.5*np.exp(-(3*x/T1)**r)+0.4*np.exp(-(6*x/T1)**r)))

        def Fit_2exponential(x, T11=0.001, T12=0.01, y0=1000, s1=1, s2=1, r=1):
            '''T1 two component exponential fit for 1/2 spin'''
            return y0*(1 -(1+s1)*np.exp(-(x/T11)**r) -(1+s2)*np.exp(-(x/T12)**r))

        def Fit_spin_7_2_TaS2(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 fit model for spin 7/2 on 1/2 - 3/2 transition; NQR'''
            return y0*(1-(1+s)*(0.024*np.exp(-(3*x/T1)**r)+0.235*np.exp(-(10*x/T1)**r)+0.741*np.exp(-(21*x/T1)**r)))
        
        def Fit_spin_7_2_TaS2_beta(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 fit model for spin 7/2 on 1/2 - 3/2 transition; NQR'''
            return y0*(1-(1+s)*(0.036*np.exp(-(2.91*x/T1)**r)+0.314*np.exp(-(9.30*x/T1)**r)+0.651*np.exp(-(19.1*x/T1)**r)))

        def Fit_spin_5_2(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 fit model for spin 5/2 on -1/2 - 1/2 transition; central line'''
            return y0*(1-(1+s)*(0.0285714*np.exp(-(1*x/T1)**r)+0.177778*np.exp(-(6*x/T1)**r)+0.793651*np.exp(-(15*x/T1)**r)))

        def Fit_spin_5_2_dbl(x, T11=0.001, T12=0.001, y0=1000, s1=1, s2=1, r=1):
            '''T1 fit model for spin 5/2 on -1/2 - 1/2 transition; central line'''
            return y0*(1-(1+s1)*(0.0285714*np.exp(-(1*x/T11)**r)+0.177778*np.exp(-(6*x/T11)**r)+0.793651*np.exp(-(15*x/T11)**r)) -(1+s2)*(0.0285714*np.exp(-(1*x/T12)**r)+0.177778*np.exp(-(6*x/T12)**r)+0.793651*np.exp(-(15*x/T12)**r)))

        def Fit_spin_5_2_1st(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 fit model for spin 5/2 on 1/2 - 3/2 transition; first sattelite'''
            return y0*(1-(1+s)*(0.0285714*np.exp(-(1*x/T1)**r)+0.0535714*np.exp(-(3*x/T1)**r)+0.025*np.exp(-(6*x/T1)**r)+0.446429*np.exp(-(10*x/T1)**r)+0.446429*np.exp(-(15*x/T1)**r)))

        def Fit_spin_5_2_1st_dbl(x, T11=0.001, T12=0.001, y0=1000, s1=1, s2=1, r=1):
            '''T1 fit model for spin 5/2 on 1/2 - 3/2 transition; first sattelite'''
            return y0*(1-(1+s1)*(0.0285714*np.exp(-(1*x/T11)**r)+0.0535714*np.exp(-(3*x/T11)**r)+0.025*np.exp(-(6*x/T11)**r)+0.446429*np.exp(-(10*x/T11)**r)+0.446429*np.exp(-(15*x/T11)**r)) -(1+s2)*(0.0285714*np.exp(-(1*x/T12)**r)+0.0535714*np.exp(-(3*x/T12)**r)+0.025*np.exp(-(6*x/T12)**r)+0.446429*np.exp(-(10*x/T12)**r)+0.446429*np.exp(-(15*x/T12)**r)))

        def Fit_spin_5_2_2nd(x, T1=0.001, y0=1000, s=1, r=1):
            '''T1 fit model for spin 5/2 on 3/2 - 5/2 transition; second sattelite'''
            return y0*(1-(1+s)*(0.0285714*np.exp(-(1*x/T1)**r)+0.214286*np.exp(-(3*x/T1)**r)+0.4*np.exp(-(6*x/T1)**r)+0.285714*np.exp(-(10*x/T1)**r)+0.0714286*np.exp(-(15*x/T1)**r)))

        def Fit_spin_5_2_2nd_dbl(x, T11=0.001, T12=0.001, y0=1000, s1=1, s2=1, r=1):
            '''T1 fit model for spin 5/2 on 3/2 - 5/2 transition; second sattelite'''
            return y0*(1-(1+s1)*(0.0285714*np.exp(-(1*x/T11)**r)+0.214286*np.exp(-(3*x/T11)**r)+0.4*np.exp(-(6*x/T11)**r)+0.285714*np.exp(-(10*x/T11)**r)+0.0714286*np.exp(-(15*x/T11)**r)) -(1+s2)*(0.0285714*np.exp(-(1*x/T12)**r)+0.214286*np.exp(-(3*x/T12)**r)+0.4*np.exp(-(6*x/T12)**r)+0.285714*np.exp(-(10*x/T12)**r)+0.0714286*np.exp(-(15*x/T12)**r)))

        def Fit_spin_5_2_2nd_tpl(x, T11=0.001, T12=0.001, T13=0.001, y0=1000, s1=1, s2=1, s3=1, r=1):
            '''T1 fit model for spin 5/2 on 3/2 - 5/2 transition; second sattelite'''
            return y0*(1-(1+s1)*(0.0285714*np.exp(-(1*x/T11)**r)+0.214286*np.exp(-(3*x/T11)**r)+0.4*np.exp(-(6*x/T11)**r)+0.285714*np.exp(-(10*x/T11)**r)+0.0714286*np.exp(-(15*x/T11)**r)) -(1+s2)*(0.0285714*np.exp(-(1*x/T12)**r)+0.214286*np.exp(-(3*x/T12)**r)+0.4*np.exp(-(6*x/T12)**r)+0.285714*np.exp(-(10*x/T12)**r)+0.0714286*np.exp(-(15*x/T12)**r)) -(1+s3)*(0.0285714*np.exp(-(1*x/T13)**r)+0.214286*np.exp(-(3*x/T13)**r)+0.4*np.exp(-(6*x/T13)**r)+0.285714*np.exp(-(10*x/T13)**r)+0.0714286*np.exp(-(15*x/T13)**r)))



        #reference to functions
        # [function, fit_params, start guess, label, tex_form]
        self.fit_names = {'Single Exp':[Fit_exponential, ['T1', 'y0', 's', 'r'],
                                        [self.trace.tau_list[self.trace.mean_range[0]-5],
                                         np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                         -self.trace.area_list[0]/self.trace.area_list[-1],
                                         1],
                                        'y0(1-(1+s)exp[-(x/T1)^r])'
                                        ],
                          'Spin 3/2':[Fit_spin_3_2, ['T1', 'y0', 's', 'r'],
                                      [6*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.1*exp(-(x/T1)**r)
+0.9*exp(-(6x/T1)**r)))'''
                                      ],
                          'Spin 3/2 double':[Fit_spin_3_2_dbl, ['T11', 'T12', 'y0', 's1', 's2', 'r'],
                                      [6*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.1*exp(-(x/T1)**r)
+0.9*exp(-(6x/T1)**r)))'''
                                      ],
                          'Spin 3/2 1st':[Fit_spin_3_2_1st, ['T1', 'y0', 's', 'r'],
                                      [6*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.1*exp(-(x/T1)**r)
+0.5*exp(-(3x/T1)**r)))
+0.4*exp(-(6x/T1)**r)))'''
                                      ],
                          'Double Exp':[Fit_2exponential, ['T11','T12','y0','s1','s2','r'],
                                        [self.trace.tau_list[self.trace.mean_range[0]-5],
                                         self.trace.tau_list[self.trace.mean_range[0]-5]*10,
                                         np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),                                    
                                         -self.trace.area_list[0]/self.trace.area_list[-1]/2,
                                         self.trace.area_list[0]/self.trace.area_list[-1]*2,
                                         1],
                                        'y0(1-(1+s1)exp[-x/T11]-(1+s2)exp[-x/T12])'
                                        ],
                          'Spin 7/2 TaS2':[Fit_spin_7_2_TaS2, ['T1', 'y0', 's', 'r'],
                                      [21*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.024*exp(-(3x/T1)**r)
+0.235*exp(-(10x/T1)**r)
+0.741*exp(-(21x/T1)**r)))'''
                                      ],
                            'Spin 7/2 TaS2 b':[Fit_spin_7_2_TaS2_beta, ['T1', 'y0', 's', 'r'],
                                      [19*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.036*exp(-(2.91x/T1)**r)
+0.314*exp(-(9.30x/T1)**r)
+0.651*exp(-(19.1x/T1)**r)))'''
                                      ],
                            'Spin 5/2':[Fit_spin_5_2, ['T1', 'y0', 's', 'r'],
                                      [15*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.029*exp(-(x/T1)**r)
+0.178*exp(-(6x/T1)**r)
+0.793*exp(-(15x/T1)**r)))'''
                                      ],
                            'Spin 5/2 double':[Fit_spin_5_2_dbl, ['T11', 'T12', 'y0', 's1', 's2', 'r'],
                                      [15*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       0.2*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -0.5*self.trace.area_list[0]/self.trace.area_list[-1],
                                       self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''sumi y0(1-(1+si)(
0.029*exp(-(x/T1i)**r)
+0.178*exp(-(6x/T1i)**r)
+0.793*exp(-(15x/T1i)**r)))'''
                                      ],
                            'Spin 5/2 1st':[Fit_spin_5_2_1st, ['T1', 'y0', 's', 'r'],
                                      [10*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.029*exp(-(x/T1)**r)
+0.054*exp(-(3x/T1)**r)
+0.025*exp(-(6x/T1)**r)
+0.446*exp(-(10x/T1)**r)
+0.446*exp(-(15x/T1)**r)))'''
                                      ],
                            'Spin 5/2 1st double':[Fit_spin_5_2_1st_dbl, ['T11', 'T12', 'y0', 's1', 's2', 'r'],
                                      [2.5*10*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       5*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -0.5*self.trace.area_list[0]/self.trace.area_list[-1],
                                       self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''sumi y0(1-(1+si)(
0.029*exp(-(x/T1i)**r)
+0.054*exp(-(3x/T1i)**r)
+0.025*exp(-(6x/T1i)**r)
+0.446*exp(-(10x/T1i)**r)
+0.446*exp(-(15x/T1i)**r)))'''
                                      ],
                            'Spin 5/2 2nd':[Fit_spin_5_2_2nd, ['T1', 'y0', 's', 'r'],
                                      [6*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''y0(1-(1+s)(
0.029*exp(-(x/T1)**r)
+0.214*exp(-(3x/T1)**r)
+0.400*exp(-(6x/T1)**r)
+0.286*exp(-(10x/T1)**r)
+0.071*exp(-(15x/T1)**r)))'''
                                      ],
                            'Spin 5/2 2nd double':[Fit_spin_5_2_2nd_dbl, ['T11', 'T12', 'y0', 's1', 's2', 'r'],
                                      [2.5*6*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       0.2*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -0.5*self.trace.area_list[0]/self.trace.area_list[-1],
                                       self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''sumi y0(1-(1+si)(
0.029*exp(-(x/T1i)**r)
+0.214*exp(-(3x/T1i)**r)
+0.400*exp(-(6x/T1i)**r)
+0.286*exp(-(10x/T1i)**r)
+0.071*exp(-(15x/T1i)**r)))'''
                                      ],
                            'Spin 5/2 2nd triple':[Fit_spin_5_2_2nd_tpl, ['T11', 'T12', 'T13', 'y0', 's1', 's2', 's3', 'r'],
                                      [2.5*6*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       0.2*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       0.05*self.trace.tau_list[self.trace.mean_range[0]-5],
                                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
                                       -0.5*self.trace.area_list[0]/self.trace.area_list[-1],
                                       self.trace.area_list[0]/self.trace.area_list[-1],
                                       self.trace.area_list[0]/self.trace.area_list[-1],
                                       1],
                                      '''sumi y0(1-(1+si)(
0.029*exp(-(x/T1i)**r)
+0.214*exp(-(3x/T1i)**r)
+0.400*exp(-(6x/T1i)**r)
+0.286*exp(-(10x/T1i)**r)
+0.071*exp(-(15x/T1i)**r)))'''
                                      ]
                         }

        def Fit():
            '''Executes the fit with given parameters and plots it'''
            Fit_function = self.fit_names[self.combo_fit_var.get()][0]
            #read values from entry boxes
            start_params = dict()
            for entry,param in zip(self.entry_params_start, param_list):
                start_params[param]=float(entry.get())
            #data points
            x = self.trace.tau_list
            y = self.trace.area_list
            y_start = [Fit_function(xx, **start_params) for xx in x]

            #check if last parameter is enabled or not
            p_start = [start_params[key] for key in param_list]
            if not self.check_params_start_var.get():
                r_tmp = p_start.pop(-1)
                self.entry_params_fit[-1].config(state='normal')
                self.entry_params_fit[-1].delete(0, 'end')
                self.entry_params_fit[-1].insert('end', r_tmp)
                self.entry_params_fit[-1].config(state='readonly')

            #run fit, p_optimal, p_covariance matrix
            if not self.check_params_start_var.get():
                popt,pcov = curve_fit(lambda x, *param_list: Fit_function(x, *param_list, r=r_tmp), x, y, p0=p_start)
            else:
                popt,pcov = curve_fit(Fit_function, x, y, p0=p_start)
            #readd last parameter
            if not self.check_params_start_var.get():
                popt = np.append(popt, r_tmp)
            y_fit = [Fit_function(xx, *popt) for xx in x]

            #print values to entry boxes
            for i,p in enumerate(popt):
                self.entry_params_fit[i].config(state='normal')
                self.entry_params_fit[i].delete(0, 'end')
                self.entry_params_fit[i].insert('end','%.4g' %p)
                self.entry_params_fit[i].config(state='readonly')

            #update plots
            self.axes_start_plot.set_ydata(y_start)
            self.axes_fit_plot.set_ydata(y_fit)
            self.fig_t1.canvas.draw()

            #save parameters
            self.trace.fit_params = popt
            self.trace.fit_param_cov = pcov
            self.trace.T1 = popt[0]
            #if self.check_params_start_var.get():
            self.trace.r = popt[-1]
            #else:
            #    self.trace.r = 1
            
            self.trace.y0 = popt[1]
            self.trace.s = popt[2]

            self.Refresh_parameters()

        def Change_fit(event=None):
            '''Changes the current fitting function'''
            #update memory in parent
            self.parent.temperatures.previous_t1['fit']=self.combo_fit_var.get()
            #repack the entries
            self.Fitting_frame()
            #rerun
            #Fit()

       
        #implement more options later if necessary
        self.label_fit = tk.Label(self.frame_fit, text='Fitting function')
        self.label_fit.pack(side='top')

        self.combo_fit_var = tk.StringVar()
        try:
            self.combo_fit_var.set(self.parent.temperatures.previous_t1['fit'])
        except KeyError:
            self.combo_fit_var.set('Single Exp')
            self.parent.temperatures.previous_t1['fit']='Single Exp'
             
        self.combo_fit = ttk.Combobox(self.frame_fit, state='readonly', values=sorted(list(self.fit_names.keys())),
                                      textvar=self.combo_fit_var)
        self.combo_fit.pack(side='top')
        self.combo_fit.bind("<<ComboboxSelected>>", Change_fit)

        

        self.label_fit_fun = tk.Label(self.frame_fit, text=self.fit_names[self.combo_fit_var.get()][3], bd=5)
        self.label_fit_fun.pack(side='top')

        self.label_starting_params = tk.Label(self.frame_fit, text='Starting values', bd=5)
        self.label_starting_params.pack(side='top')

        param_list = self.fit_names[self.combo_fit_var.get()][1]
        #guesses for where params should start
        start_guess = self.fit_names[self.combo_fit_var.get()][2]
##        start_guess = [self.trace.tau_list[self.trace.mean_range[0]-5],
##                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
##                       -self.trace.area_list[0]/self.trace.area_list[-1],
##                       1]
##        if self.combo_fit_var.get() == 'Spin 3/2':
##            start_guess[0] = start_guess[0]*6

        #start parameters entry rows
        self.frame_params_start = list()
        self.label_params_start = list()
        self.entry_params_start = list()
        for i,param in enumerate(param_list):
            self.frame_params_start.append(tk.Frame(self.frame_fit))
            self.frame_params_start[i].pack(side='top', fill='y')

            self.label_params_start.append(tk.Label(self.frame_params_start[i], text=param+' = '))
            self.label_params_start[i].pack(side='left', anchor='e')

            self.entry_params_start.append(tk.Entry(self.frame_params_start[i],
                                                    width=10, justify='right'))
            self.entry_params_start[i].insert(0, '%.4g' % start_guess[i])
            self.entry_params_start[i].pack(side='left', anchor='e')

        #check button for stretch
        self.check_params_start_var = tk.BooleanVar(self, 0)
        try:
            if self.trace.r != 1:
                self.check_params_start_var.set(1)
        except AttributeError: pass
        self.check_params_start = (ttk.Checkbutton(self.frame_params_start[-1],
                                                   variable=self.check_params_start_var))
        self.check_params_start.pack(side='left')
            
        self.button_fit = ttk.Button(self.frame_fit, text='Retry fit', command=Fit)
        self.button_fit.pack(side='top')

        self.label_fit_params = tk.Label(self.frame_fit, text='Fitted values', bd=5)
        self.label_fit_params.pack(side='top')

        #fit results entry rows
        self.frame_params_fit = list()
        self.label_params_fit = list()
        self.entry_params_fit = list()
        for i,param in enumerate(param_list):
            self.frame_params_fit.append(tk.Frame(self.frame_fit))
            self.frame_params_fit[i].pack(side='top', fill='y')

            self.label_params_fit.append(tk.Label(self.frame_params_fit[i], text=param+' = '))
            self.label_params_fit[i].pack(side='left')

            self.entry_params_fit.append(tk.Entry(self.frame_params_fit[i], width=10,
                                                  state='readonly', justify='right'))
            self.entry_params_fit[i].pack(side='left')

        #run first lap of fit
        Fit()

        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_fit, text='Confirm', command=self.Confirm)
        self.button_confirm.pack(side='bottom')
        self.button_confirm.bind('<Return>', self.Confirm)

        #add export csv button
        self.button_export = ttk.Button(self.frame_fit, text='Export CSV', command=self.Export)
        self.button_export.pack(side='bottom')
        self.button_export.bind('<F5>', self.Export)

    def Confirm(self, event=None):
        '''Confirm the selection in this screen'''
        #unpack, dont destroy untill series is done, in case corrections are needed
        self.parent.temperatures.wait.set(False)
        self.pack_forget()

        self.destroy()


        #move to later stages
        self.trace.analysed = True

    def Refresh_parameters(self):
        '''refreshes the parameters table'''
        self.tree_parameters.delete(*self.tree_parameters.get_children())
        self.trace.Get_params()
        for item in GLOBAL_t1_displayed_params:
            try:
                pair = (item, self.trace.__dict__[item])
                self.tree_parameters.insert('', 'end', values=pair)
            except: pass

    def Export(self, event=None):
        '''Saves the datapoints of the plot to a CSV file'''
        file_name = self.trace.file_key  + '.csv'
        file_directory = os.path.join('data', self.parent.current_experiment, 'csv', 'T1_raw', )

        #make the csv folder for old experiments
        try:
            os.mkdir(file_directory)
        except: pass

        #write file
        with open(os.path.join(file_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            #name row
            writer.writerow(['tau(s)', 'Signal(a.u.)'])
            #data
            for i in range(len(self.trace.tau_list)):
                row = [self.trace.tau_list[i], self.trace.area_list[i]]
                writer.writerow(row)

        tk.messagebox.showinfo('Export complete', 'The file was saved as '+file_name)

    def Fill_plot(self):
        '''Plots the T1 trend and fits it'''
        #data lines
        x = self.trace.tau_list
        y = self.trace.area_list

        #T1 plot
        self.axes = self.fig_t1.add_subplot(111)
        self.axes.plot(x, y, 'bo', color=colors[1], label='Data')
        self.axes_start_plot, = self.axes.plot(x, y, color=colors[3],
                                               linestyle='dashed', label='Fit start')
        self.axes_fit_plot, = self.axes.plot(x, y, color=colors[4], label='Fit')
        self.axes.axvline(x=x[self.trace.mean_range[0]], color=colors[2])
        self.axes.set_xscale('log')
        self.axes.set_title('T1')
        self.axes.set_xlabel(r'$\tau$ (s)')
        self.axes.set_ylabel('Signal')
        legend = self.axes.legend(loc='lower right')
        self.axes.grid()

class Frame_plot_T1_t1vt(tk.Frame):
    '''T1vT trend plotting'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent)
        self.pack(side='left', anchor='n')
        
        #reference to parent
        self.parent = parent
        #reference to current series
        self.trace = trace

        #counter for plots
        self.counter = 0
        
        #load widgets
        self.Widgets()

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #button commands
        def Confirm(event=None):
            '''Confirm the selection in this screen'''
            #unpack, dont destroy untill series is done, in case corrections are needed
            self.parent.temperatures.wait.set(False)
            self.pack_forget()
            self.destroy()
            plt.close('all')

            self.parent.traces.button_show.config(state='normal')

        #split in columns
        self.frame_plot_left = tk.Frame(self)
        self.frame_plot_left.pack(side='left', anchor='n')
        self.frame_plot_right = tk.Frame(self)
        self.frame_plot_right.pack(side='left', anchor='n')

        #plot frames
        self.frame_plot1 = tk.Frame(self.frame_plot_left, bd=5)
        self.frame_plot1.pack(side='top', anchor='n')
        self.frame_plot2 = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_plot2.pack(side='top', anchor='n')
        self.frame_plot3 = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_plot3.pack(side='top', anchor='n')

        #buttons frame
        self.frame_buttons = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_buttons.pack(side='top', anchor='e')
        
        #T1 plot
        self.fig_t1vt = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_t1vt.subplots_adjust(bottom=0.12, left= 0.11, right=0.96, top=0.94)
        self.canvas_t1vt = FigureCanvasTkAgg(self.fig_t1vt, self.frame_plot1)
        self.canvas_t1vt._tkcanvas.pack()

        #fr plot
        self.fig_fr = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_fr.subplots_adjust(bottom=0.18, left= 0.11, right=0.96, top=0.90)
        self.canvas_fr = FigureCanvasTkAgg(self.fig_fr, self.frame_plot2)
        self.canvas_fr._tkcanvas.pack()

        #stretch plot
        self.fig_r = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_r.subplots_adjust(bottom=0.18, left= 0.11, right=0.96, top=0.90)
        self.canvas_r = FigureCanvasTkAgg(self.fig_r, self.frame_plot3)
        self.canvas_r._tkcanvas.pack()


        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_buttons, text='Confirm', command=Confirm)
        self.button_confirm.pack(side='right')
        self.button_confirm.bind('<Return>', Confirm)

        #plot the stuff
        self.Fill_plot()

        #add button to export parameters
        self.button_export = ttk.Button(self.frame_buttons, text='Export CSV', command=self.Export)
        self.button_export.pack(side='right')



    def Add_trace(self, trace):
        '''Adds traces to plots using given trace'''
        #initialize lists
        x = list()
        y = list()
        fr = list()
        r = list()
        y0 = list()
        s = list()
        dT1 = list()

        #prepare all items for export
        popts = list()
        pcovs = list()
        
        for temp in self.trace:
            if self.trace[temp].analysed and not self.trace[temp].disabled:

                x.append(temp)
                y.append(self.trace[temp].T1)
                #maby calculate some center frequency at some point?
                fr.append(self.trace[temp].fr)
                #get stretch
                try:
                    r.append(self.trace[temp].r)
                except AttributeError:
                    r.append(1)
                #get y0 and s if exist
                try:
                    y0.append(self.trace[temp].y0)
                    s.append(self.trace[temp].s)
                except AttributeError:
                    pass

                dT1.append(np.sqrt(self.trace[temp].fit_param_cov[0][0]))
                
                popts.append(self.trace[temp].fit_params)
                pcovs.append(self.trace[temp].fit_param_cov)
                
                name = self.trace[temp].file_key
        #sort by temperature
        sorting = np.argsort(x)
        x = np.array(x)[sorting]
        y = np.array(y)[sorting]
        y2 = 1/y
        fr = np.array(fr)[sorting]
        r = np.array(r)[sorting]
        y0 = np.array(y0)[sorting]
        s = np.array(s)[sorting]
        dT1 = np.array(dT1)[sorting]
        popts = np.array(popts)[sorting]
        pcovs = np.array(pcovs)[sorting]
        

        #draw trace
        self.axes_1.plot(x, y2, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')
        self.axes_2.plot(x, fr, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')
        self.axes_3.plot(x, r, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')

        #save for export
        self.data = dict()
        self.data['T'] = x
        self.data['T1'] = y
        self.data['fr'] = fr
        self.data['r'] = r
        self.data['y0'] = y0
        self.data['s'] = s
        self.data['dT1'] = dT1
        self.data['popts'] = popts
        self.data['pcovs'] = pcovs

        #increase plot counter
        self.counter += 1

    def Export(self):
        '''Saves the plotted data into a CSV file for further analysis'''
        file_name = self.parent.current_trace + '.csv'
        file_directory = os.path.join('data', self.parent.current_experiment, 'csv', 'T1')

        #make the csv folder for old experiments
        try:
            os.mkdir(file_directory)
        except: pass

        #write file
        with open(os.path.join(file_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            #name row
            writer.writerow(['T(K)', 'T1(s)', 'fr(MHz)', 'r', 'y0', 's', 'dT1'])
            #data
            for i in range(len(self.data['T'])):
                row = [self.data['T'][i], self.data['T1'][i], self.data['fr'][i], self.data['r'][i],
                       self.data['y0'][i], self.data['s'][i], self.data['dT1'][i]] + list(self.data['popts'][i])
                writer.writerow(row)


        tk.messagebox.showinfo('Export complete', 'The file was saved as '+file_name)

    def Fill_plot(self):
        '''Creates the plots for T1vT'''

        self.axes_1 = self.fig_t1vt.add_subplot(111)
        self.axes_1.set_xscale('log')
        self.axes_1.set_yscale('log')
        self.axes_1.set_title('T1 temperature dependence')
        self.axes_1.set_xlabel('Temperature (K)')
        self.axes_1.set_ylabel(r'1/T1 (1/s)')
        #self.axes_1.legend(loc='lower right')
        self.axes_1.grid()

        self.axes_2 = self.fig_fr.add_subplot(111)
        self.axes_2.set_title('Center frequencies')
        self.axes_2.set_xlabel('Temperature (K)')
        self.axes_2.set_ylabel('Frequency (MHz)')
        #self.axes_2.get_yaxis().get_major_formatter().set_useOffset(False)
        self.axes_2.margins(0.05, 0.1)
        self.axes_2.grid()

        self.axes_3 = self.fig_r.add_subplot(111)
        self.axes_3.set_title('Stretch')
        self.axes_3.set_xlabel('Temperature (K)')
        self.axes_3.set_ylabel('Stretch r')
        #self.axes_3.get_yaxis().get_major_formatter().set_useOffset(False)
        self.axes_3.margins(0.05, 0.1)
        self.axes_3.grid()


class Frame_plot_T2_quick(tk.Frame):
    '''Pioneer first T2 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='both', expand=True)
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace

        #starting data
        self.range = self.parent.temperatures.previous_t2['mean_range'][1]

        #load widgets
        self.Widgets()

        #run quick t2
        quick_tables = trace.Quick_T2()
        self.Fill_plots(*quick_tables)

        #take focus away from listbox
        self.focus()
        #global key binds
        root.bind('<Left>', self.Interrupt)
        root.bind('<Right>', self.Finish)
        


    def Finish(self, event=None):
        '''Accepts the data on this screen and closes it up'''
        #save data
        self.trace.mean_range = (0, self.range)
        self.trace.mean_shl = int(self.mean_shl)
        self.trace.mean_phase = self.mean_phase
        self.parent.temperatures.previous_t2['mean_range']=(0, self.range)
        #hide this frame
        self.pack_forget()
        #close plots
        plt.close('all')
        #forget global key bind
        root.unbind('<Right>')
        root.unbind('<Left>')

        #run next frame
        self.parent.plot_t2_ranges = Frame_plot_T2_ranges(self.parent, self.trace)

    def Interrupt(self, event=None):
        '''Stops the analysis loop'''
        #Destroy frame and plots
        self.pack_forget()
        self.destroy()
        plt.close('all')
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')

        #stop the analysis loop
        self.parent.temperatures.wait.set(False)

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #split in two half frames
        self.frame_left = tk.Frame(self)
        self.frame_right = tk.Frame(self)
        self.frame_left.pack(side='left', fill='y')
        self.frame_right.pack(side='left', fill='y')

        #add frames on left side
        self.frame_left1 = tk.Frame(self.frame_left, bd=5)
        self.frame_left2 = tk.Frame(self.frame_left, bd=5)
        self.frame_left3 = tk.Frame(self.frame_left, bd=5)
        self.frame_left1.pack(side='top')
        self.frame_left2.pack(side='top')
        self.frame_left3.pack(side='top', fill='x')
        #add frames on right side
        self.frame_right1 = tk.Frame(self.frame_right, bd=5)
        self.frame_right2 = tk.Frame(self.frame_right, bd=5)
        self.frame_right1.pack(side='top')
        self.frame_right2.pack(side='top')

        #add canvases and toolbars
        #plot 1
        self.fig_left1 = plt.figure(dpi=100, figsize=(7,3))
        self.fig_left1.subplots_adjust(bottom=0.20, left= 0.12, right=0.96, top=0.88)
        self.fig_left1.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')
        self.canvas_left1 = FigureCanvasTkAgg(self.fig_left1, self.frame_left1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left1, self.frame_left1)
        self.canvas_left1._tkcanvas.pack()

        #plot 2
        self.fig_left2 = plt.figure(dpi=100, figsize=(7,4))
        self.fig_left2.subplots_adjust(bottom=0.15, left= 0.12, right=0.96, top=0.9)
        self.canvas_left2 = FigureCanvasTkAgg(self.fig_left2, self.frame_left2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left2, self.frame_left2)
        self.canvas_left2._tkcanvas.pack()
        
        #interrupt button
        self.button_interrupt = ttk.Button(self.frame_left3, text='Interrupt', command=self.Interrupt)
        self.button_interrupt.pack(side='left', anchor='w')

        #label and edit of mean_range
        self.frame_left3_middle = tk.Frame(self.frame_left3)
        self.frame_left3_middle.pack(anchor='center')

        self.label_mean = tk.Label(self.frame_left3_middle,  text='Selected range:')
        self.label_mean.pack(side='left')

        self.entry_mean_var = tk.StringVar(self, value=self.range)
        self.entry_mean = ttk.Entry(self.frame_left3_middle,
                                    textvariable=self.entry_mean_var, width=3)
        self.entry_mean.pack(side='left')


        #plot 3
        self.fig_right1 = plt.figure(dpi=100, figsize=(7,3.5))
        self.fig_right1.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.9)
        self.canvas_right1 = FigureCanvasTkAgg(self.fig_right1, self.frame_right1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right1, self.frame_right1)
        self.canvas_right1._tkcanvas.pack()

        #plot 4
        self.fig_right2 = plt.figure(dpi=100, figsize=(7,3.5))
        self.fig_right2.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.9)
        self.canvas_right2 = FigureCanvasTkAgg(self.fig_right2, self.frame_right2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right2, self.frame_right2)
        self.canvas_right2._tkcanvas.pack()


        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_right, text='Confirm', command=self.Finish)
        self.button_confirm.pack(side='top', anchor='ne')

    def Fill_plots(self, temp_list, temp_list2, temp_set, tau_list, t2_list, phase_list, shl_list):
        '''Puts the contents into the plot fields'''
        #starting values
        self.mean_t2 = np.mean(t2_list[:self.range])
        self.mean_phase = np.mean(np.unwrap(phase_list[:self.range]))
        self.mean_shl = np.round(np.mean(shl_list[:self.range]))
        #x axes
        n = len(tau_list)
        x_list = np.linspace(1,n,n)
        
        #plot 1, temperature stabillity    
        self.axes_left1 = self.fig_left1.add_subplot(111)
        if abs(np.mean(temp_list) - temp_set) < 2:
            self.axes_left1.plot(x_list, temp_list, marker='.', color=colors[1], label='ITC_R1')
        if abs(np.mean(temp_list2) - temp_set) < 2:
            self.axes_left1.plot(x_list, temp_list2, marker='.', color=colors[2], label='ITC_R2')
        self.axes_left1.axhline(y=temp_set, color=colors[0], label='Set T')
        self.axes_left1.margins(0.02, 0.1)
        self.axes_left1.set_title('Temperature stabillity check')
        self.axes_left1.set_xlabel('File index')
        self.axes_left1.set_ylabel('Temperature (K)')
        self.axes_left1.legend(loc='upper right')
        self.axes_left1.grid()

        #plot 2
        self.axes_left2 = self.fig_left2.add_subplot(111)
        self.axes_left2.plot(tau_list, t2_list, 'bo', color=colors[1], label='Data')
        self.axes_left2_vline = self.axes_left2.axvline(x=tau_list[self.range],
                                                        color=colors[2], label='Select')
        self.axes_left2_hline = self.axes_left2.axhline(y=self.mean_t2, color=colors[0],
                                                        label='Plato')
        self.axes_left2.set_yscale('log')
        self.axes_left2.set_title('T2 quick check')
        self.axes_left2.set_xlabel(r'$\tau$ ($\mu$s)')
        self.axes_left2.set_ylabel('Signal')
        #legend = self.axes_left2.legend(loc='lower right')
        #legend.draggable()
        self.axes_left2.grid()
        
        #plot 3
        self.axes_right1 = self.fig_right1.add_subplot(111)
        self.axes_right1.plot(x_list, np.unwrap(np.array(phase_list))*180/np.pi, marker='.',
                             color=colors[1], label='Phase')
        self.axes_right1_hline = self.axes_right1.axhline(self.mean_phase*180/np.pi, color=colors[0], label='Mean phase')
        self.axes_right1.margins(0.02, 0.1)
        self.axes_right1.set_title('Phase check')
        self.axes_right1.set_xlabel('File index')
        self.axes_right1.set_ylabel('Phase (Deg)')
        #self.axes_right1.legend(loc='lower right')
        self.axes_right1.grid()

        #plot 4
        self.axes_right2 = self.fig_right2.add_subplot(111)
        self.axes_right2.plot(x_list, shl_list, marker='.',
                             color=colors[1], label='SHL')
        self.axes_right2_hline = self.axes_right2.axhline(self.mean_shl,
                                                          color=colors[0], label='Mean SHL')
        self.axes_right2.margins(0.02, 0.1)
        self.axes_right2.set_title('SHL check')
        self.axes_right2.set_xlabel('File index')
        self.axes_right2.set_ylabel('Shift left')
        #self.axes_right2.legend(loc='lower right')
        self.axes_right2.grid()

        #redraw canvases
        self.fig_left1.canvas.draw()
        self.fig_left2.canvas.draw()
        self.fig_right1.canvas.draw()
        self.fig_right2.canvas.draw()
        

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, recalculates mean of selected points'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.range = np.searchsorted(tau_list, event.xdata, side='right')
                self.mean_t2 = np.mean(t2_list[:self.range])
                self.mean_phase = np.mean(np.unwrap(phase_list[:self.range]))
                self.mean_shl = np.round(np.mean(shl_list[:self.range]))
                self.entry_mean_var.set(self.range)
                #update plot
                self.axes_left2_vline.set_xdata(event.xdata)
                self.axes_left2_hline.set_ydata(self.mean_t2)
                self.axes_right1_hline.set_ydata(self.mean_phase*180/np.pi)
                self.axes_right2_hline.set_ydata(self.mean_shl)
                self.fig_left2.canvas.draw()
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()

        self.axes_left2_vline_drag = self.fig_left2.canvas.mpl_connect('motion_notify_event', Drag)

class Frame_plot_T2_ranges(tk.Frame):
    '''Pioneer first T2 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='both', expand=True)
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace

        self.offset_select = self.parent.temperatures.previous_t2['offset'][0]
        self.range_l_select = self.parent.temperatures.previous_t2['integral_range'][0]
        self.range_r_select = self.parent.temperatures.previous_t2['integral_range'][1]
        self.mirroring = self.parent.temperatures.previous_t2['mirroring']
        
        #load widgets
        self.Widgets()

        #load plots and read
        self.Choose_offset(trace)

        self.focus()
        #global key bindings
        root.bind('<Left>', self.Previous)
        root.bind('<Right>', self.Confirm_offset)

        
   

    def Widgets(self):
        '''Builds all the subframes and canvases'''
     
        def Set_offset(event=None):
            '''Entry change of offset, replot and write value'''
            try:
                self.offset_select = int(self.entry_offset.get())
                #update plot
                self.axes_left1_vline.set_xdata(self.offset_select)
                self.axes_left2_vline.set_xdata(self.offset_select)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted values must be integers!')

        def Set_range(event=None):
            '''Entry change of ranges, replot and save value'''
            try:
                self.range_l_select = int(self.entry_range_l_var.get())
                self.range_r_select = int(self.entry_range_r_var.get())
                self.axes_right1_vline_l.set_xdata(self.spc_fr[self.range_l_select])
                self.axes_right2_vline_l.set_xdata(self.spc_fr[self.range_l_select])
                self.axes_right1_vline_r.set_xdata(self.spc_fr[self.range_r_select])
                self.axes_right2_vline_r.set_xdata(self.spc_fr[self.range_r_select])
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted values must be integers!')
            

        #split in two half frames
        self.frame_left = tk.Frame(self)
        self.frame_right = tk.Frame(self)
        self.frame_left.pack(side='left', fill='y')
        self.frame_right.pack(side='left', fill='y')

        #add frames on left side
        self.frame_left1 = tk.Frame(self.frame_left, bd=5)
        self.frame_left2 = tk.Frame(self.frame_left, bd=5)
        self.frame_left3 = tk.Frame(self.frame_left, bd=5)
        self.frame_left1.pack(side='top')
        self.frame_left2.pack(side='top')
        self.frame_left3.pack(side='top', fill='x')
        #add frames on right side
        self.frame_right1 = tk.Frame(self.frame_right, bd=5)
        self.frame_right2 = tk.Frame(self.frame_right, bd=5)
        self.frame_right3 = tk.Frame(self.frame_right, bd=5)
        self.frame_right1.pack(side='top')
        self.frame_right2.pack(side='top')
        self.frame_right3.pack(side='top', fill='x')

        #add canvases and toolbars
        #plot 1
        self.fig_left1 = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_left1.subplots_adjust(bottom=0.20, left= 0.10, right=0.96, top=0.88)
        self.fig_left1.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')
        self.canvas_left1 = FigureCanvasTkAgg(self.fig_left1, self.frame_left1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left1, self.frame_left1)
        self.canvas_left1._tkcanvas.pack()

        #plot 2
        self.fig_left2 = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_left2.subplots_adjust(bottom=0.12, left= 0.08, right=0.96, top=0.93)
        self.canvas_left2 = FigureCanvasTkAgg(self.fig_left2, self.frame_left2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left2, self.frame_left2)
        self.canvas_left2._tkcanvas.pack()
 
        #buttons left
        self.button_previous = ttk.Button(self.frame_left3, text='Repeat previous', command=self.Previous)
        self.button_previous.pack(side='left')

        self.button_confirm = ttk.Button(self.frame_left3, text='Confirm', command=self.Confirm_offset)
        self.button_confirm.pack(side='right')

        #check button for mirroring fid
        self.check_mirroring_var = tk.BooleanVar(self, False)
        if self.mirroring:
            self.check_mirroring_var.set(True)
            
        self.check_mirroring = (ttk.Checkbutton(self.frame_left3, variable=self.check_mirroring_var))
        self.check_mirroring.pack(side='right')
        
        self.label_mirroring = tk.Label(self.frame_left3,  text='Mirroring')
        self.label_mirroring.pack(side='right')
        

        #middle frame
        self.frame_left3_middle = tk.Frame(self.frame_left3)
        self.frame_left3_middle.pack(anchor='center')

        self.label_offset = tk.Label(self.frame_left3_middle,  text='Selected offset:')
        self.label_offset.pack(side='left')
        
        self.entry_offset_var = tk.StringVar(self, value=self.offset_select)
        self.entry_offset = ttk.Entry(self.frame_left3_middle,
                                      textvariable=self.entry_offset_var, width=5)
        self.entry_offset.pack(side='left')
        self.entry_offset.bind('<Return>', Set_offset)

        self.button_set_offset = ttk.Button(self.frame_left3_middle,
                                            text='Set offset', command=Set_offset)
        self.button_set_offset.pack(side='left')


        #plot 3
        self.fig_right1 = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_right1.subplots_adjust(bottom=0.20, left= 0.10, right=0.96, top=0.88)
        self.canvas_right1 = FigureCanvasTkAgg(self.fig_right1, self.frame_right1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right1, self.frame_right1)
        self.canvas_right1._tkcanvas.pack()

        #plot 4
        self.fig_right2 = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_right2.subplots_adjust(bottom=0.12, left= 0.10, right=0.96, top=0.93)
        self.canvas_right2 = FigureCanvasTkAgg(self.fig_right2, self.frame_right2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right2, self.frame_right2)
        self.canvas_right2._tkcanvas.pack()

        #buttons right
        self.label_range = tk.Label(self.frame_right3,  text='Selected ranges:')
        self.label_range.pack(side='left')
        
        self.entry_range_l_var = tk.StringVar(self, value=self.range_l_select)
        self.entry_range_l = ttk.Entry(self.frame_right3,
                                       textvariable=self.entry_range_l_var, width=5)
        self.entry_range_l.pack(side='left')
        self.entry_range_l.bind('<Return>', Set_range)

        self.label_range_comma = tk.Label(self.frame_right3,  text=' , ')
        self.label_range_comma.pack(side='left')

        self.entry_range_r_var = tk.StringVar(self, value=self.range_r_select)
        self.entry_range_r = ttk.Entry(self.frame_right3,
                                       textvariable=self.entry_range_r_var, width=5)
        self.entry_range_r.pack(side='left')
        self.entry_range_r.bind('<Return>', Set_range)

        self.button_set_range = ttk.Button(self.frame_right3, text='Set range', command=Set_range)
        self.button_set_range.pack(side='left')

        self.button_close = ttk.Button(self.frame_right3, text='Confirm',
                                       command=self.Close, state='disabled')
        self.button_close.pack(side='right')

    #button commands
    def Previous(self, event=None):
        '''Back to the previous step!'''
        #reload offset
        self.parent.plot_t2_quick.pack(side='left', fill='both', expand=True)
        #destroy me
        self.pack_forget()
        self.destroy()
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')


    def Confirm_offset(self, event=None):
        '''Saves current offset range and opens integral ranges select'''
        self.trace.offset_range = (self.offset_select, None)
        self.parent.temperatures.previous_t2['offset'] = (self.offset_select, None)
        
        #remember mirroring
        self.parent.temperatures.previous_t2['mirroring'] = self.check_mirroring_var.get()
        self.trace.mirroring = self.check_mirroring_var.get()
        
        self.Choose_ranges(self.trace)
        self.button_confirm.config(state='disabled')
        self.button_close.config(state='enabled')
        self.button_close.focus_set()

        #change global keys
        root.bind('<Right>', self.Close)

    def Close(self, event=None):
        '''Confirm the selection in this screen'''
        #save the integral ranges
        self.trace.integral_range = (self.range_l_select, self.range_r_select)
        self.parent.temperatures.previous_t2['integral_range'] = (self.range_l_select,
                                                                  self.range_r_select)

        #finish the analysis
        self.trace.Run()
        
        #unpack and destroy
        self.trace.analysed = True
        self.parent.plot_t2_quick.destroy()
        self.pack_forget()
        self.destroy()
        plt.close('all')
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')

        #load the overview frame
        self.parent.plot_t2_view = Frame_plot_T2_view(self.parent, self.trace)

    def Choose_offset(self, trace):
        '''Operations and plotting for choosing the FID offsets'''
        fids = list()
        for file in trace.file_list:
            fid = FID(file, trace.file_dir)
            fids.append(fid.x)

        x_mean = np.mean(fids[slice(*trace.mean_range)], axis=0)

        #plot 1
        self.axes_left1 = self.fig_left1.add_subplot(111)
        self.axes_left1.plot(np.real(x_mean), color=colors[1], label='Re')
        self.axes_left1.plot(np.imag(x_mean), color=colors[2], label='Im')
        self.axes_left1.plot(np.abs(x_mean), color=colors[0], label='Abs')
        self.axes_left1.axvline(x=trace.mean_shl, color=colors[-1])
        self.axes_left1_vline = self.axes_left1.axvline(x=self.offset_select, color=colors[4])
        self.axes_left1.margins(0.02, 0.1)
        self.axes_left1.set_title('Mean FID')
        self.axes_left1.set_xlabel('Time (index)')
        self.axes_left1.set_ylabel('Signal (A.U.)')
        #self.axes_left1.legend(loc='upper right')
        self.axes_left1.grid()

        #plot 2
        self.axes_left2 = self.fig_left2.add_subplot(111, sharex=self.axes_left1)
        for i, fid in enumerate(fids):
            self.axes_left2.plot(np.abs(fid)+np.amax(np.abs(x_mean))*0.5*i,
                                 color=colors[i%9], label=str(i))
        self.axes_left2.axvline(x=trace.mean_shl, color=colors[-1], label='shl')
        self.axes_left2_vline = self.axes_left2.axvline(x=self.offset_select,
                                                        color=colors[4], label='Select')
        self.axes_left2.margins(0.02, 0.02)
        self.axes_left2.set_title('All FIDs')
        self.axes_left2.set_xlabel('Time (index)')
        self.axes_left2.set_ylabel('Absolute signal (A.U.)')
        self.axes_left2.grid()

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, recalculates mean of selected points'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.offset_select = int(event.xdata)
                self.entry_offset_var.set(self.offset_select)
                #update plot
                self.axes_left1_vline.set_xdata(event.xdata)
                self.axes_left2_vline.set_xdata(event.xdata)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()

        self.axes_left1_vline_drag = self.fig_left1.canvas.mpl_connect('motion_notify_event', Drag)
        self.axes_left2_vline_drag = self.fig_left2.canvas.mpl_connect('motion_notify_event', Drag)

    def Choose_ranges(self, trace):
        '''Operations and plotting for choosing spectrum integral ranges'''
        spcs = list()
        for file in trace.file_list:
            fid = FID(file, trace.file_dir)
            fid.Offset(trace.offset_range)
            fid.Shift_left(trace.mean_shl, mirroring=trace.mirroring)
            fid.Fourier()
            fid.Phase_rotate(trace.mean_phase)

            spcs.append(fid.spc)

        spc_fr = fid.spc_fr
        self.spc_fr = spc_fr
        spc_mean = np.mean(spcs[slice(*trace.mean_range)], axis=0)

        #plot 3
        self.axes_right1 = self.fig_right1.add_subplot(111)
        self.axes_right1.plot(spc_fr, np.real(spc_mean), color=colors[1], label='Re')
        self.axes_right1.plot(spc_fr, np.imag(spc_mean), color=colors[2], label='Im')
        self.axes_right1.axvline(x=trace.fr, color=colors[-1])
        self.axes_right1_vline_l = self.axes_right1.axvline(x=spc_fr[self.range_l_select],
                                                            color=colors[4])
        self.axes_right1_vline_r = self.axes_right1.axvline(x=spc_fr[self.range_r_select],
                                                            color=colors[4])
        self.axes_right1.set_xlim((trace.fr -0.5,+ trace.fr +0.5))
        self.axes_right1.set_title('Mean spectrum (Drag with left and right mouse button)')
        self.axes_right1.set_xlabel('Frequency (MHz)')
        self.axes_right1.set_ylabel('Signal (A.U.)')
        self.axes_right1.legend(loc='upper left')
        self.axes_right1.grid()

        #plot 4
        self.axes_right2 = self.fig_right2.add_subplot(111)
        for i, spc in enumerate(spcs):
            self.axes_right2.plot(spc_fr, np.real(spc)+np.amax(np.abs(spc_mean))*0.5*i,
                                  color=colors[i%9], label=str(i))
        self.axes_right1.axvline(x=trace.fr, color=colors[-1])
        self.axes_right2_vline_l = self.axes_right2.axvline(x=spc_fr[self.range_l_select],
                                                            color=colors[4])
        self.axes_right2_vline_r = self.axes_right2.axvline(x=spc_fr[self.range_r_select],
                                                            color=colors[4])
        self.axes_right2.set_xlim((trace.fr -0.5,+ trace.fr +0.5))
        self.axes_right2.margins(0.02, 0.02)
        self.axes_right2.set_title('All FIDs')
        self.axes_right2.set_xlabel('Frequency (MHz)')
        self.axes_right2.set_ylabel('Real part of signal (A.U.)')
        self.axes_right2.grid()

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, recalculates mean of selected points'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.range_l_select = np.searchsorted(spc_fr, event.xdata, side='left')
                self.entry_range_l_var.set(self.range_l_select)
                #update plot
                self.axes_right1_vline_l.set_xdata(event.xdata)
                self.axes_right2_vline_l.set_xdata(event.xdata)
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()

            if event.button == 3 and event.inaxes != None:
                #find the index of selected points
                self.range_r_select = np.searchsorted(spc_fr, event.xdata, side='right')
                self.entry_range_r_var.set(self.range_r_select)
                #update plot
                self.axes_right1_vline_r.set_xdata(event.xdata)
                self.axes_right2_vline_r.set_xdata(event.xdata)
                self.fig_right1.canvas.draw()
                self.fig_right2.canvas.draw()

        self.axes_right1_vline_drag = self.fig_right1.canvas.mpl_connect('motion_notify_event', Drag)
        self.axes_right2_vline_drag = self.fig_right2.canvas.mpl_connect('motion_notify_event', Drag)

        self.fig_right1.canvas.draw()
        self.fig_right2.canvas.draw()
        
class Frame_plot_T2_view(tk.Frame):
    '''Pioneer first T2 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent)
        self.pack(side='left', anchor='n')
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace
        
        #load widgets
        self.Widgets()

        #global key bind
        root.bind('<Right>', self.Confirm)

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #button commands

        def Disable(event=None):
            '''Disables and red-flags the point to avoid plotting'''
            #try:
            self.trace.disabled = not self.trace.disabled
            #except:
            #    self.trace.disabled = True
            self.Refresh_parameters()

        def Repeat(event=None):
            '''Clears the T2 trace and starts the analysis from scratch'''
            self.trace.Reinit()
            self.Confirm()
            self.trace.analysed = False

            
        #bottom button row
        self.frame_bottom = tk.Frame(self)
        self.frame_bottom.pack(side='bottom', fill='x')
        #split in columns
        self.frame_parameters = tk.Frame(self, bd=5)
        self.frame_parameters.pack(side='left', anchor='n')
        self.frame_plot = tk.Frame(self, bd=5)
        self.frame_plot.pack(side='left', anchor='n')


        #parameters
        self.label_parameters = tk.Label(self.frame_parameters,  text='Parameters')
        self.label_parameters.pack(side='top')

        self.tree_columns = ('Name','Value')
        self.tree_parameters = ttk.Treeview(self.frame_parameters, columns=self.tree_columns,
                                            show='headings', selectmode='none', height=25)
        self.tree_parameters.pack(side='top',fill='y', expand=True)

        #define column widths
        self.tree_parameters.column('Name', width=80)
        self.tree_parameters.column('Value', width=120)
        #define column names
        for column in self.tree_columns:
            self.tree_parameters.heading(column, text=column)
        #display in degrees
        self.trace.mean_phase_deg = self.trace.mean_phase*180/np.pi
        #fill in params
        self.Refresh_parameters()
        
        # disable point button
        self.button_disable = ttk.Button(self.frame_parameters, text='Disable/enable Point',
                                         command=Disable, width=20)
        self.button_disable.pack(side='top')
        #redo analysis button
        self.button_repeat = ttk.Button(self.frame_parameters, text='Repeat analysis',
                                        command=Repeat, width=20)
        self.button_repeat.pack(side='top')

        #T2 plot
        self.fig_t2 = plt.figure(dpi=100, figsize=(8,6))
        self.fig_t2.subplots_adjust(bottom=0.1, left= 0.10, right=0.96, top=0.94)
        self.fig_t2.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')
        self.fig_t2.text(0.82, 0.97, r'$y_0 \exp(-(\frac{x}{T_2})^r)$',
                                horizontalalignment='center', verticalalignment='center')


        self.canvas_t2 = FigureCanvasTkAgg(self.fig_t2, self.frame_plot)
        self.canvas_t2._tkcanvas.pack()

        self.Fill_plot()
        self.Fitting_frame()
        
    def Fitting_frame(self, event=None):
        '''Initializes/repacks the fitting frame, depending on the selected fitting function'''
        #repack
        try:
            self.frame_fit.destroy()
        except:
            pass
        #fit frame
        self.frame_fit = tk.Frame(self, bd=5)
        self.frame_fit.pack(side='left', anchor='n', fill='y')

        #fit functions
        def Fit_exponential(x, T2=0.001, y0=1000, r=1):
            '''T2 exponential fit model'''
            return y0*np.exp(-(x/T2)**r)

        def Fit_2exponential(x, T21=0.001, T22=0.01, y01=100, y02=100, r=1):
            '''T2 double exponential decay model'''
            return y01*np.exp(-x/T21) + y02*np.exp(-x/T22)\

        def Fit_exp_gauss(x, T21=0.001, T22=0.01, y01=100, y02=100, r=1):
            '''T2 exponential plus gaussian model'''
            return y01*np.exp(-x/T21) + y02*np.exp(-(x/T22)**2)

        def Fit_exp_gauss_2(x, T21=0.001, T22=0.01, y0=100, r=1):
            '''T2 exponential and gaussian model'''
            return y0*np.exp(-x/T21 -(x/(2*T22))**2)

        #fit start point estimates
        yy1=self.trace.area_list[0]
        yy2=self.trace.area_list[5]
        xx1=self.trace.tau_list[0]
        xx2=self.trace.tau_list[5]
        kk=(yy2-yy1)/(xx2-xx1)
        nn=yy1-kk*xx1

        #reference to functions   #[self.trace.tau_list[self.trace.mean_range[1]]
        # [function, fit_params, start guess, label, tex_form]

        self.fit_names = {'Single Exp':[Fit_exponential, ['T2', 'y0', 'r'], [-nn/kk, nn, 1],
                                        'y0*exp(-(x/T2)^r)'],
                          'Double Exp':[Fit_2exponential, ['T21','T22', 'y01', 'y02', 'r'],
                                        [-nn/kk, -nn/kk*10, nn/2, nn/2, 1],
                                        'y01*exp(-x/T21)+y02*exp(-x/T22)'],
                          'Gauss Exp':[Fit_exp_gauss, ['T21','T22', 'y01', 'y02', 'r'],
                                        [-nn/kk, -nn/kk, nn/2, nn/2, 1],
                                        'y01*exp(-x/T21)+y02*exp((-x/T22)**2)'],
                          'Gauss Exp 2':[Fit_exp_gauss_2, ['T21','T22', 'y0', 'r'],
                                        [-nn/kk, -nn/kk, nn/2, 1],
                                        'y0*exp(-x/T21 -(x/(2*T22))**2 )']
                         }

        
        def Fit():
            '''Executes the fit with given parameters and plots it'''
            Fit_function = self.fit_names[self.combo_fit_var.get()][0]
            #read values from entry boxes
            start_params = dict()
            for entry,param in zip(self.entry_params_start, param_list):
                start_params[param]=float(entry.get())
            #data points
            x = np.array(self.trace.tau_list)*2
            y = self.trace.area_list
            y_start = [Fit_function(xx, **start_params) for xx in x]

            #check if last parameter is enabled or not
            p_start = [start_params[key] for key in param_list]
            if not self.check_params_start_var.get():
                r_tmp = p_start.pop(-1)
                self.entry_params_fit[-1].config(state='normal')
                self.entry_params_fit[-1].delete(0, 'end')
                self.entry_params_fit[-1].insert('end', 1)
                self.entry_params_fit[-1].config(state='readonly')

            #run fit, p_optimal, p_covariance matrix
            if not self.check_params_start_var.get():
                popt,pcov = curve_fit(lambda x, *param_list: Fit_function(x, *param_list, r=r_tmp), x, y, p0=p_start)
            else:
                popt,pcov = curve_fit(Fit_function, x, y, p0=p_start)
            #readd last parameter
            if not self.check_params_start_var.get():
                popt = np.append(popt, r_tmp)
            y_fit = [Fit_function(xx, *popt) for xx in x]

            #print values to entry boxes
            for i,p in enumerate(popt):
                self.entry_params_fit[i].config(state='normal')
                self.entry_params_fit[i].delete(0, 'end')
                self.entry_params_fit[i].insert('end','%.4g' %p)
                self.entry_params_fit[i].config(state='readonly')

            #update plots
            self.axes_start_plot.set_ydata(y_start)
            self.axes_fit_plot.set_ydata(y_fit)
            self.fig_t2.canvas.draw()

            #save parameters
            self.trace.fit_params = popt
            self.trace.fit_param_cov = pcov
            self.trace.fit_fun = self.combo_fit_var.get()
            self.trace.T2 = popt[0]
            #if self.check_params_start_var.get():
            self.trace.r = popt[-1]
            #else:
            #    self.trace.r = 1

            self.Refresh_parameters()

        def Change_fit(event=None):
            '''Changes the current fitting function'''
            #update memory in parent
            self.parent.temperatures.previous_t2['fit']=self.combo_fit_var.get()
            #reload
            self.Fitting_frame()
            #rerun
            #Fit()

       
        #implement more options later if necessary
        self.label_fit = tk.Label(self.frame_fit, text='Fitting function')
        self.label_fit.pack(side='top')

        self.combo_fit_var = tk.StringVar()
        try:
            self.combo_fit_var.set(self.parent.temperatures.previous_t2['fit'])
        except KeyError:
            self.combo_fit_var.set('Single Exp')
            self.parent.temperatures.previous_t2['fit']='Single Exp'
             
        self.combo_fit = ttk.Combobox(self.frame_fit, state='readonly', values=sorted(list(self.fit_names.keys())),
                                      textvar=self.combo_fit_var)
        self.combo_fit.pack(side='top')
        self.combo_fit.bind("<<ComboboxSelected>>", Change_fit)

        

        self.label_fit_fun = tk.Label(self.frame_fit, text=self.fit_names[self.combo_fit_var.get()][3], bd=5)
        self.label_fit_fun.pack(side='top')

        self.label_starting_params = tk.Label(self.frame_fit, text='Starting values', bd=5)
        self.label_starting_params.pack(side='top')

        param_list = self.fit_names[self.combo_fit_var.get()][1]
        #guesses for where params should start
        start_guess = self.fit_names[self.combo_fit_var.get()][2]

        #start parameters entry rows
        self.frame_params_start = list()
        self.label_params_start = list()
        self.entry_params_start = list()
        for i,param in enumerate(param_list):
            self.frame_params_start.append(tk.Frame(self.frame_fit))
            self.frame_params_start[i].pack(side='top', fill='y')

            self.label_params_start.append(tk.Label(self.frame_params_start[i], text=param+' = '))
            self.label_params_start[i].pack(side='left', anchor='e')

            self.entry_params_start.append(tk.Entry(self.frame_params_start[i],
                                                    width=10, justify='right'))
            self.entry_params_start[i].insert(0, '%.4g' % start_guess[i])
            self.entry_params_start[i].pack(side='left', anchor='e')

        self.check_params_start_var = tk.BooleanVar(self, 0)
        self.check_params_start = (ttk.Checkbutton(self.frame_params_start[-1],
                                                   variable=self.check_params_start_var))
        self.check_params_start.pack(side='left')
            
        self.button_fit = ttk.Button(self.frame_fit, text='Retry fit', command=Fit)
        self.button_fit.pack(side='top')

        self.label_fit_params = tk.Label(self.frame_fit, text='Fitted values', bd=5)
        self.label_fit_params.pack(side='top')

        #fit results entry rows
        self.frame_params_fit = list()
        self.label_params_fit = list()
        self.entry_params_fit = list()
        for i,param in enumerate(param_list):
            self.frame_params_fit.append(tk.Frame(self.frame_fit))
            self.frame_params_fit[i].pack(side='top', fill='y')

            self.label_params_fit.append(tk.Label(self.frame_params_fit[i], text=param+' = '))
            self.label_params_fit[i].pack(side='left')

            self.entry_params_fit.append(tk.Entry(self.frame_params_fit[i], width=10,
                                                  state='readonly', justify='right'))
            self.entry_params_fit[i].pack(side='left')

        #run first lap of fit
        Fit()

        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_fit, text='Confirm', command=self.Confirm)
        self.button_confirm.pack(side='bottom')
        self.button_confirm.bind('<Return>', self.Confirm)

        #export csv button
        self.button_export = ttk.Button(self.frame_fit, text='Export CSV', command=self.Export)
        self.button_export.pack(side='bottom')
        self.button_export.bind('<F5>', self.Export)

    def Confirm(self, event=None):
        '''Confirm the selection in this screen'''
        #unpack, dont destroy untill series is done, in case corrections are needed
        self.parent.temperatures.wait.set(False)
        self.pack_forget()
        self.destroy()
        plt.close('all')

        #move to later stages
        self.trace.analysed = True

    def Refresh_parameters(self):
        '''refreshes the parameters table'''
        self.tree_parameters.delete(*self.tree_parameters.get_children())
        self.trace.Get_params()
        for item in GLOBAL_t2_displayed_params:
            try:
                pair = (item, self.trace.__dict__[item])
                self.tree_parameters.insert('', 'end', values=pair)
            except: pass

    def Export(self, event=None):
        '''Saves the datapoints of the plot to a CSV file'''
        file_name = self.trace.file_key  + '.csv'
        file_directory = os.path.join('data', self.parent.current_experiment, 'csv', 'T2_raw', )

        #make the csv folder for old experiments
        try:
            os.mkdir(file_directory)
        except: pass

        #write file
        with open(os.path.join(file_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            #name row
            writer.writerow(['tau(s)', 'Signal(a.u.)'])
            #data
            for i in range(len(self.trace.tau_list)):
                row = [self.trace.tau_list[i], self.trace.area_list[i]]
                writer.writerow(row)

        tk.messagebox.showinfo('Export complete', 'The file was saved as '+file_name)

    def Fill_plot(self):
        '''Plots the T1 trend and fits it'''
        #data lines
        x = np.array(self.trace.tau_list)*2
        y = np.array(self.trace.area_list)

        #T1 plot
        self.axes = self.fig_t2.add_subplot(111)
        self.axes.plot(x, y, 'bo', color=colors[1], label='Data')
        self.axes_start_plot, = self.axes.plot(x, y, color=colors[3],
                                               linestyle='dashed', label='Fit start')
        self.axes_fit_plot, = self.axes.plot(x, y, color=colors[4], label='Fit')
        self.axes.axvline(x=x[self.trace.mean_range[1]], color=colors[2])
        self.axes.set_yscale('log')
        self.axes.set_title('T2')
        self.axes.set_xlabel(r'2*$\tau$ ($\mu$s)')
        self.axes.set_ylabel('Signal')
        legend = self.axes.legend(loc='lower right')
        self.axes.grid()


class Frame_plot_T2_t2vt(tk.Frame):
    '''T2vT trend plotting'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent)
        self.pack(side='left', anchor='n')
        
        #reference to parent
        self.parent = parent
        #reference to current series
        self.trace = trace

        #counter for plots
        self.counter = 0
        
        #load widgets
        self.Widgets()

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #button commands
        def Confirm(event=None):
            '''Confirm the selection in this screen'''
            #unpack, dont destroy untill series is done, in case corrections are needed
            self.parent.temperatures.wait.set(False)
            self.pack_forget()
            self.destroy()
            plt.close('all')

            self.parent.traces.button_show.config(state='normal')

        #split in columns
        self.frame_plot_left = tk.Frame(self)
        self.frame_plot_left.pack(side='left', anchor='n')
        self.frame_plot_right = tk.Frame(self)
        self.frame_plot_right.pack(side='left', anchor='n')

        #plot frames
        self.frame_plot1 = tk.Frame(self.frame_plot_left, bd=5)
        self.frame_plot1.pack(side='top', anchor='n')
        self.frame_plot2 = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_plot2.pack(side='top', anchor='n')
        self.frame_plot3 = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_plot3.pack(side='top', anchor='n')

        #buttons frame
        self.frame_buttons = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_buttons.pack(side='top', anchor='e')
        
        #T1 plot
        self.fig_t2vt = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_t2vt.subplots_adjust(bottom=0.12, left= 0.11, right=0.96, top=0.94)
        self.canvas_t2vt = FigureCanvasTkAgg(self.fig_t2vt, self.frame_plot1)
        self.canvas_t2vt._tkcanvas.pack()

        #fr plot
        self.fig_fr = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_fr.subplots_adjust(bottom=0.18, left= 0.11, right=0.96, top=0.90)
        self.canvas_fr = FigureCanvasTkAgg(self.fig_fr, self.frame_plot2)
        self.canvas_fr._tkcanvas.pack()

        #stretch plot
        self.fig_r = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_r.subplots_adjust(bottom=0.18, left= 0.11, right=0.96, top=0.90)
        self.canvas_r = FigureCanvasTkAgg(self.fig_r, self.frame_plot3)
        self.canvas_r._tkcanvas.pack()


        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_buttons, text='Confirm', command=Confirm)
        self.button_confirm.pack(side='right', anchor='e')
        self.button_confirm.bind('<Return>', Confirm)

        self.Fill_plot()

        #add button to export to csv
        self.button_export = ttk.Button(self.frame_buttons, text='Export CSV', command=self.Export)
        self.button_export.pack(side='right', anchor='e')

    def Add_trace(self, trace):
        '''Adds traces to plots using given trace'''
        #initialize lists
        x = list()
        y = list()
        fr = list()
        r = list()

        #prepare all items for export
        popts = list()
        pcovs = list()

      
        
        for temp in self.trace:
            if self.trace[temp].analysed and not self.trace[temp].disabled:

                x.append(temp)
                y.append(self.trace[temp].T2)
                #maby calculate some center frequency at some point?
                fr.append(self.trace[temp].fr)
                #get stretch
                try:
                    r.append(self.trace[temp].r)
                except AttributeError:
                    r.append(1)
                #get second T2
                except AttributeError:
                    yb.append(1)
                name = self.trace[temp].file_key

                popts.append(self.trace[temp].fit_params)
                pcovs.append(self.trace[temp].fit_param_cov)
                
        #sort by temperature
        sorting = np.argsort(x)
        x = np.array(x)[sorting]
        y = np.array(y)[sorting]
        y2 = 1/y
        fr = np.array(fr)[sorting]
        r = np.array(r)[sorting]
        popts = np.array(popts)[sorting]
        pcovs = np.array(pcovs)[sorting]

        #draw trace
        self.axes_1.plot(x, y2, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')
        self.axes_2.plot(x, fr, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')
        self.axes_3.plot(x, r, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')


        #save for export
        self.data = dict()
        self.data['T'] = x
        self.data['T2'] = y
        self.data['fr'] = fr
        self.data['r'] = r
        self.data['popts'] = popts
        self.data['pcovs'] = pcovs

        #increase plot counter
        self.counter += 1

    def Export(self):
        '''Saves the plotted data into a CSV file for further analysis'''
        file_name = self.parent.current_trace + '.csv'
        file_directory = os.path.join('data', self.parent.current_experiment, 'csv', 'T2')

        #make the csv folder for old experiments
        try:
            os.mkdir(file_directory)
        except: pass

        #write file
        with open(os.path.join(file_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            #name row
            writer.writerow(['T(K)', 'T2(s)', 'fr(MHz)', 'r', 'popts'])
            #data
            for i in range(len(self.data['T'])):
                row = [self.data['T'][i], self.data['T2'][i], self.data['fr'][i], self.data['r'][i]] + list(self.data['popts'][i])
                writer.writerow(row)

        tk.messagebox.showinfo('Export complete', 'The file was saved as '+file_name)

    def Fill_plot(self):
        '''Creates the plots for T2vT'''

        self.axes_1 = self.fig_t2vt.add_subplot(111)
        self.axes_1.set_xscale('log')
        self.axes_1.set_yscale('log')
        self.axes_1.set_title('T2 temperature dependence')
        self.axes_1.set_xlabel('Temperature (K)')
        self.axes_1.set_ylabel(r'1/T2 (1/$\mu$s)')
        #self.axes_1.legend(loc='lower right')
        self.axes_1.grid()

        self.axes_2 = self.fig_fr.add_subplot(111)
        self.axes_2.set_title('Center frequencies')
        self.axes_2.set_xlabel('Temperature (K)')
        self.axes_2.set_ylabel('Frequency (MHz)')
        #self.axes_2.get_yaxis().get_major_formatter().set_useOffset(False)
        self.axes_2.margins(0.05, 0.1)
        self.axes_2.grid()

        self.axes_3 = self.fig_r.add_subplot(111)
        self.axes_3.set_title('Stretch')
        self.axes_3.set_xlabel('Temperature (K)')
        self.axes_3.set_ylabel('Stretch r')
        #self.axes_3.get_yaxis().get_major_formatter().set_useOffset(False)
        self.axes_3.margins(0.05, 0.1)
        self.axes_3.grid()


class Frame_plot_spc_quick(tk.Frame):
    '''Quick settings for spectrum gluing'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='both', expand=True)
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace

        #starting data
        self.shl = self.parent.temperatures.previous_spc['shl_start']
        self.mirroring = self.parent.temperatures.previous_spc['mirroring']
        self.offset_select = self.parent.temperatures.previous_spc['offset'][0]

        #load widgets
        self.Widgets()

        #run quick t1
        quick_tables = trace.Quick_spc()
        self.Fill_plots(*quick_tables)

        #take focus away from listbox
        self.focus()
        #global key binds
        root.bind('<Left>', self.Interrupt)
        root.bind('<Right>', self.Confirm)


    def Confirm(self, event=None):
        '''Saves shl, disables its selection and plots phases'''
        #disable shl setters
        self.entry_shl.config(state='disabled')
        self.button_set_shl.config(state='disabled')
        self.check_mirroring.config(state='disabled')
        self.button_confirm_shl.config(state='disabled')
        self.entry_offset.config(state='disabled')
        self.button_set_offset.config(state='disabled')
        #disconnect drag events
        self.fig_left2.canvas.mpl_disconnect(self.axes_left2_vline_drag)
        self.fig_right1.canvas.mpl_disconnect(self.axes_right1_hline_drag)

        #enable confirm button
        self.button_confirm.config(state='enabled')
        self.entry_k.config(state='enabled')
        self.entry_n.config(state='enabled')
        self.button_fit.config(state='enabled')

        #remember shl
        self.mean_shl = self.shl
        self.trace.mean_shl = self.shl
        self.parent.temperatures.previous_spc['shl_start']=self.shl
        
        #remember offset
        self.trace.offset_range = (self.offset_select, None)
        self.parent.temperatures.previous_spc['offset'] = (self.offset_select, None)
        
        #remember mirroring
        self.trace.mirroring = self.check_mirroring_var.get()
        self.parent.temperatures.previous_spc['mirroring'] = self.check_mirroring_var.get()


        self.trace.Get_phase(self.shl, (self.offset_select,None),
                             (self.fit_range_l_var.get(), self.fit_range_r_var.get())
                             )

        
        #update entry values
        self.entry_k_var.set('%.4g' %self.trace.phase_fit_p[0])
        self.entry_n_var.set('%.4g' %self.trace.phase_fit_p[1])


        #plot 4 phase
        self.axes_right2 = self.fig_right2.add_subplot(111)
        self.axes_right2.plot(self.trace.fr_list, np.unwrap(self.trace.phase_list, 0.5*np.pi), marker='.',
                             color=colors[1], label='SHL')
        self.axes_right2_line2, = self.axes_right2.plot(self.trace.fr_list, self.trace.phase_fit,
                                                        color=colors[2], label='linear fit')
        self.axes_vline_l = self.axes_right2.axvline(x=self.trace.fr_list[self.fit_range_l_var.get()], color=colors[4])
        self.axes_vline_r = self.axes_right2.axvline(x=self.trace.fr_list[self.fit_range_r_var.get()-1], color=colors[4])
        self.axes_right2.margins(0.02, 0.1)
        self.axes_right2.set_title('Phase check')
        self.axes_right2.set_xlabel('Frequency (MHz)')
        self.axes_right2.set_ylabel('Shift left')
        #self.axes_right2.legend(loc='lower right')
        self.axes_right2.grid()
        self.fig_right2.canvas.draw()

        def Drag(event):
            '''Allows dragging of the markers for fit range on spectrum plot'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.fit_range_l_var.set(np.searchsorted(self.trace.fr_list, event.xdata, side='right'))
                #print(self.fit_range_l_var.get())
                #self.range_l_select = int(event.xdata)
                #update plot
                self.axes_vline_l.set_xdata(event.xdata)
                self.fig_right2.canvas.draw()

            if event.button == 3 and event.inaxes != None:
                #find the index of selected points
                self.fit_range_r_var.set(np.searchsorted(self.trace.fr_list, event.xdata, side='left'))
                #print(self.fit_range_r_var.get())
                #self.range_r_select = int(event.xdata)
                #update plot
                self.axes_vline_r.set_xdata(event.xdata)
                self.fig_right2.canvas.draw()

        self.axes_vline_drag = self.fig_right2.canvas.mpl_connect('motion_notify_event', Drag)

        root.bind('<Right>', self.Finish)


        
    def Finish(self, event=None):
        '''Accepts the data on this screen and closes it up'''      

        #hide this frame
        self.pack_forget()
        #close plots
        plt.close('all')
        #forget global key bind
        root.unbind('<Right>')
        root.unbind('<Left>')

        #run next frame
        self.parent.plot_spc_ranges = Frame_plot_spc_ranges(self.parent, self.trace)

    def Interrupt(self, event=None):
        '''Stops the analysis loop'''
        #Destroy frame and plots
        self.pack_forget()
        self.destroy()
        plt.close('all')
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')

        #stop the analysis loop
        self.parent.temperatures.wait.set(False)

    def Refit(self, event=None):
        '''manual fit, change k, n values'''

##        #manual version
##        k = float(self.entry_k_var.get())
##        n = float(self.entry_n_var.get())
##        x = np.array(self.trace.fr_list)
##        y = k*x+n
##        self.axes_right2_line2.set_ydata(y)
##
##        self.fig_right2.canvas.draw()
##
##        self.trace.phase_fit_p = [k, n]
##        self.trace.phase_fir = y

        #refit using new fit range
        self.trace.Get_phase(self.shl, (self.offset_select,None),
                             (self.fit_range_l_var.get(), self.fit_range_r_var.get())
                             )

        self.axes_right2_line2.set_ydata(self.trace.phase_fit)
        self.fig_right2.canvas.draw()

        self.entry_k_var.set('%.4g' %self.trace.phase_fit_p[0])
        self.entry_n_var.set('%.4g' %self.trace.phase_fit_p[1])


    def Widgets(self):
        '''Builds all the subframes and canvases'''

        def Set_shl(event=None):
            '''Entry chang eof shl, replot with new value'''
            try:
                self.shl = int(self.entry_shl.get())
                #update plots
                self.axes_left2_vline.set_xdata(self.shl)
                self.axes_right1_hline.set_ydata(self.shl)
                self.fig_left2.canvas.draw()
                self.fig_right1.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted value must be integer!')

        def Set_offset(event=None):
            '''Entry change of offset select, replot with new value'''
            try:
                self.offset_select = int(self.entry_offset.get())
                #update plots
                self.axes_left2_vline_offset.set_xdata(self.offset_select)
                self.fig_left2.canvas.draw()

            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted value must be integer!')
        
        #split in two half frames
        self.frame_left = tk.Frame(self)
        self.frame_right = tk.Frame(self)
        self.frame_left.pack(side='left', fill='y')
        self.frame_right.pack(side='left', fill='y')

        #add frames on left side
        self.frame_left1 = tk.Frame(self.frame_left, bd=5)
        self.frame_left2 = tk.Frame(self.frame_left, bd=5)
        self.frame_left3 = tk.Frame(self.frame_left, bd=5)
        self.frame_left1.pack(side='top')
        self.frame_left2.pack(side='top')
        self.frame_left3.pack(side='top', fill='x')
        #add frames on right side
        self.frame_right1 = tk.Frame(self.frame_right, bd=5)
        self.frame_right2 = tk.Frame(self.frame_right, bd=5)
        self.frame_right3 = tk.Frame(self.frame_right, bd=5)
        self.frame_right4 = tk.Frame(self.frame_right, bd=5)
        self.frame_right1.pack(side='top')
        self.frame_right2.pack(side='top', fill='x')
        self.frame_right3.pack(side='top')
        self.frame_right4.pack(side='top', fill='x')

        

        #add canvases and toolbars
        #plot 1
        self.fig_left1 = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_left1.subplots_adjust(bottom=0.20, left= 0.12, right=0.96, top=0.88)
        self.fig_left1.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')
        self.canvas_left1 = FigureCanvasTkAgg(self.fig_left1, self.frame_left1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left1, self.frame_left1)
        self.canvas_left1._tkcanvas.pack()

        #plot 2
        self.fig_left2 = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_left2.subplots_adjust(bottom=0.15, left= 0.12, right=0.96, top=0.9)
        self.canvas_left2 = FigureCanvasTkAgg(self.fig_left2, self.frame_left2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left2, self.frame_left2)
        self.canvas_left2._tkcanvas.pack()
        
        #interrupt button
        self.button_interrupt = ttk.Button(self.frame_left3, text='Interrupt', command=self.Interrupt)
        self.button_interrupt.pack(side='left', anchor='w')

        #confirm shl selection, jump to phase
        self.button_confirm_shl = ttk.Button(self.frame_left3, text='Confirm', command=self.Confirm)
        self.button_confirm_shl.pack(side='right', anchor='e')
        
        #check button for mirroring fid
        self.check_mirroring_var = tk.BooleanVar(self, False)
        if self.mirroring:
            self.check_mirroring_var.set(True)
            
        self.check_mirroring = (ttk.Checkbutton(self.frame_left3, variable=self.check_mirroring_var))
        self.check_mirroring.pack(side='right')
        
        self.label_mirroring = tk.Label(self.frame_left3,  text='Mirroring')
        self.label_mirroring.pack(side='right')

        #label and edit of mean_range
        self.frame_left3_middle = tk.Frame(self.frame_left3)
        self.frame_left3_middle.pack(anchor='center')

        self.label_offset = tk.Label(self.frame_left3_middle,  text='Chosen offset range:')
        self.label_offset.pack(side='left')

        self.entry_offset_var = tk.StringVar(self, value=self.offset_select)
        self.entry_offset = ttk.Entry(self.frame_left3_middle,
                                    textvariable=self.entry_offset_var, width=4)
        self.entry_offset.pack(side='left')
        self.entry_offset.bind('<Return>', Set_offset)

        self.button_set_offset = ttk.Button(self.frame_left3_middle, text='Set offset', command=Set_offset)
        self.button_set_offset.pack(side='left')

        #plot 3
        self.fig_right1 = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_right1.subplots_adjust(bottom=0.20, left= 0.12, right=0.96, top=0.88)
        self.canvas_right1 = FigureCanvasTkAgg(self.fig_right1, self.frame_right1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right1, self.frame_right1)
        self.canvas_right1._tkcanvas.pack()

        #label and edit of shl select
        self.frame_right2_middle = tk.Frame(self.frame_right2)
        self.frame_right2_middle.pack(anchor='center')

        self.label_shl = tk.Label(self.frame_right2_middle,  text='Chosen shl:')
        self.label_shl.pack(side='left')

        self.entry_shl_var = tk.StringVar(self, value=self.shl)
        self.entry_shl = ttk.Entry(self.frame_right2_middle,
                                    textvariable=self.entry_shl_var, width=4)
        self.entry_shl.pack(side='left')
        self.entry_shl.bind('<Return>', Set_shl)

        self.button_set_shl = ttk.Button(self.frame_right2_middle, text='Set SHL', command=Set_shl)
        self.button_set_shl.pack(side='left')



        #plot 4
        self.fig_right2 = plt.figure(dpi=100, figsize=(7,4))
        self.fig_right2.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.9)
        self.canvas_right2 = FigureCanvasTkAgg(self.fig_right2, self.frame_right3)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_right2, self.frame_right2)
        self.canvas_right2._tkcanvas.pack()

        #entries for phase fit (lin)
        self.label_k = tk.Label(self.frame_right4, text='k:')
        self.label_k.pack(side='left')

        self.entry_k_var = tk.StringVar(self, value=None)
        self.entry_k = ttk.Entry(self.frame_right4, textvariable=self.entry_k_var, width=6, state='disabled')
        self.entry_k.pack(side='left')

        self.label_n = tk.Label(self.frame_right4, text='n:')
        self.label_n.pack(side='left')

        self.entry_n_var = tk.StringVar(self, value=None)
        self.entry_n = ttk.Entry(self.frame_right4, textvariable=self.entry_n_var, width=6, state='disabled')
        self.entry_n.pack(side='left')

        self.button_fit = ttk.Button(self.frame_right4, text='Refit', command=self.Refit, state='disabled')
        self.button_fit.pack(side='left')

        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_right4, text='Confirm', command=self.Finish, state='disabled')
        self.button_confirm.pack(side='right')


        #fitting range variables for phase fit
        self.fit_range_l_var = tk.IntVar(self, value=0)
        self.fit_range_r_var = tk.IntVar(self, value=-1)
        try:
            self.fit_range_l_var.set(self.trace.fit_range[0])
            self.fit_range_r_var.set(self.trace.fit_range[1])
        except:
            pass

       

    def Fill_plots(self, temp_list, temp_list2, temp_set, fr_list, shl_list):
        '''Puts the contents into the plot fields'''
        #starting values
        self.mean_shl = np.mean(shl_list)

        #x axes
        n = len(fr_list)
        x_list = np.linspace(1,n,n)

        #fids
        fids = list()
        for file in self.trace.file_list:
            fid = FID(file, self.trace.file_dir)
            fids.append(fid.x)
        
        #plot 1, temperature stabillity    
        self.axes_left1 = self.fig_left1.add_subplot(111)
        try:
            if abs(np.mean(temp_list) - temp_set) < 2:
                self.axes_left1.plot(x_list, temp_list, marker='.', color=colors[1], label='ITC_R1')
            if abs(np.mean(temp_list2) - temp_set) < 2:
                self.axes_left1.plot(x_list, temp_list2, marker='.', color=colors[2], label='ITC_R2')
        except: pass
        self.axes_left1.axhline(y=temp_set, color=colors[0], label='Set T')
        self.axes_left1.margins(0.02, 0.1)
        self.axes_left1.set_title('Temperature stabillity check')
        self.axes_left1.set_xlabel('File index')
        self.axes_left1.set_ylabel('Temperature (K)')
        self.axes_left1.legend(loc='upper right')
        self.axes_left1.grid()

        #plot 2
        self.axes_left2 = self.fig_left2.add_subplot(111)
        for i, fid in enumerate(fids):
            self.axes_left2.plot(np.abs(fid)+np.amax(np.abs(fids))*0.5*i,
                                 color=colors[i%9], label=str(i))
        self.axes_left2_vline = self.axes_left2.axvline(x=self.shl,
                                                        color=colors[0], label='Select')
        self.axes_left2_vline_offset = self.axes_left2.axvline(x=self.offset_select,
                                                        color=colors[2], label='Offset Select')
        self.axes_left2.set_title('offset range select')
        self.axes_left2.set_xlabel('Time, (A.U.)')
        self.axes_left2.set_ylabel('Signal')
        #legend = self.axes_left2.legend(loc='lower right')
        #legend.draggable()
        self.axes_left2.grid()
        
        #plot 3
        self.axes_right1 = self.fig_right1.add_subplot(111)
        self.axes_right1.plot(x_list, shl_list, marker='.',
                             color=colors[1], label='SHL')
        self.axes_right1_hline = self.axes_right1.axhline(self.shl,
                                                          color=colors[0], label='Mean SHL')
        self.axes_right1.margins(0.02, 0.1)
        self.axes_right1.set_title('SHL select')
        self.axes_right1.set_xlabel('File index')
        self.axes_right1.set_ylabel('Shift left')
        #self.axes_right1.legend(loc='lower right')
        self.axes_right1.grid()

        #redraw canvases
        self.fig_left1.canvas.draw()
        self.fig_left2.canvas.draw()
        self.fig_right1.canvas.draw()
        

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, redraws the line on right'''
            if event.button == 3 and event.inaxes != None:
                #find the index of selected points
                self.offset_select = int(event.xdata)
                self.entry_offset_var.set(self.offset_select)
                #update plot
                self.axes_left2_vline_offset.set_xdata(event.xdata)
                self.fig_left2.canvas.draw()

        #draggable vline event
        def Drag_shl(event):
            '''Allows dragging of the marker in left2, redraws the line on right'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.shl = int(event.xdata)
                self.entry_shl_var.set(self.shl)
                #update plot
                self.axes_left2_vline.set_xdata(self.shl)
                self.axes_right1_hline.set_ydata(self.shl)
                self.fig_left2.canvas.draw()
                self.fig_right1.canvas.draw()


        #draggable hline event
        def Drag_shl2(event):
            ''''Allows dragging of the marker in right1, redraws shl lines'''
            if event.button == 1 and event.inaxes !=None:
                #find selected index
                self.shl = int(event.ydata)
                self.entry_shl_var.set(self.shl)
                #update plots
                self.axes_left2_vline.set_xdata(self.shl)
                self.axes_right1_hline.set_ydata(event.ydata)
                self.fig_left2.canvas.draw()
                self.fig_right1.canvas.draw()
                

        self.axes_left2_vline_drag = self.fig_left2.canvas.mpl_connect('motion_notify_event', Drag)
        self.axes_left2_vline_drag = self.fig_left2.canvas.mpl_connect('motion_notify_event', Drag_shl)
        self.axes_right1_hline_drag = self.fig_right1.canvas.mpl_connect('motion_notify_event', Drag_shl2)


class Frame_plot_spc_ranges(tk.Frame):
    '''Pioneer first T2 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent, bd=5)
        self.pack(side='left', fill='both', expand=True)
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace

        self.range_l_select = self.parent.temperatures.previous_spc['integral_range'][0]
        self.range_r_select = self.parent.temperatures.previous_spc['integral_range'][1]
        
        #load widgets
        self.Widgets()

        #load plots and read
        self.Choose_ranges(trace)

        self.focus()
        #global key bindings
        root.bind('<Left>', self.Previous)
        root.bind('<Right>', self.Finish)

        
   

    def Widgets(self):
        '''Builds all the subframes and canvases'''
     
        def Set_offset(event=None):
            '''Entry change of offset, replot and write value'''
            try:
                self.offset_select = int(self.entry_offset.get())
                #update plot
                self.axes_left1_vline.set_xdata(self.offset_select)
                self.axes_left2_vline.set_xdata(self.offset_select)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted values must be integers!')

        def Set_range(event=None):
            '''Entry change of ranges, replot and save value'''
            try:
                self.range_l_select = int(self.entry_range_l_var.get())
                self.range_r_select = int(self.entry_range_r_var.get())
                self.axes_left1_vline_l.set_xdata(self.spc_fr[self.range_l_select])
                self.axes_left2_vline_l.set_xdata(self.spc_fr[self.range_l_select])
                self.axes_left1_vline_r.set_xdata(self.spc_fr[self.range_r_select])
                self.axes_left2_vline_r.set_xdata(self.spc_fr[self.range_r_select])
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()
            except ValueError:
                tk.messagebox.showerror('Error', 'The inserted values must be integers!')
            

        #split in two half frames
        self.frame_left = tk.Frame(self)
        self.frame_right = tk.Frame(self)
        self.frame_left.pack(side='left', fill='y')
        self.frame_right.pack(side='left', fill='y')

        #add frames on left side
        self.frame_left1 = tk.Frame(self.frame_left, bd=5)
        self.frame_left2 = tk.Frame(self.frame_left, bd=5)
        self.frame_left3 = tk.Frame(self.frame_left, bd=5)
        self.frame_left1.pack(side='top')
        self.frame_left2.pack(side='top')
        self.frame_left3.pack(side='top', fill='x')
        #add frames on right side
        self.frame_right1 = tk.Frame(self.frame_right, bd=5)
        self.frame_right2 = tk.Frame(self.frame_right, bd=5)
        self.frame_right3 = tk.Frame(self.frame_right, bd=5)
        self.frame_right1.pack(side='top')
        self.frame_right2.pack(side='top')
        self.frame_right3.pack(side='top', fill='x')

        #plot 1
        self.fig_left1 = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_left1.subplots_adjust(bottom=0.20, left= 0.10, right=0.96, top=0.88)
        self.canvas_left1 = FigureCanvasTkAgg(self.fig_left1, self.frame_left1)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left1, self.frame_left1)
        self.canvas_left1._tkcanvas.pack()

        #plot 2
        self.fig_left2 = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_left2.subplots_adjust(bottom=0.12, left= 0.10, right=0.96, top=0.93)
        self.canvas_left2 = FigureCanvasTkAgg(self.fig_left2, self.frame_left2)
        #self.toolbar = NavigationToolbar2TkAgg(self.canvas_left2, self.frame_left2)
        self.canvas_left2._tkcanvas.pack()

        #buttons left
        self.button_previous = ttk.Button(self.frame_left3, text='Repeat previous', command=self.Previous)
        self.button_previous.pack(side='left')

        self.button_confirm = ttk.Button(self.frame_left3, text='Confirm', command=self.Finish)
        self.button_confirm.pack(side='right')

        #middle frame
        self.frame_left3_middle = tk.Frame(self.frame_left3)
        self.frame_left3_middle.pack(anchor='center')

        self.label_range = tk.Label(self.frame_left3_middle,  text='Selected ranges:')
        self.label_range.pack(side='left')
        
        self.entry_range_l_var = tk.StringVar(self, value=self.range_l_select)
        self.entry_range_l = ttk.Entry(self.frame_left3_middle,
                                       textvariable=self.entry_range_l_var, width=5)
        self.entry_range_l.pack(side='left')
        self.entry_range_l.bind('<Return>', Set_range)

        self.label_range_comma = tk.Label(self.frame_left3_middle,  text=' , ')
        self.label_range_comma.pack(side='left')

        self.entry_range_r_var = tk.StringVar(self, value=self.range_r_select)
        self.entry_range_r = ttk.Entry(self.frame_left3_middle,
                                       textvariable=self.entry_range_r_var, width=5)
        self.entry_range_r.pack(side='left')
        self.entry_range_r.bind('<Return>', Set_range)

        self.button_set_range = ttk.Button(self.frame_left3_middle, text='Set range', command=Set_range)
        self.button_set_range.pack(side='left')



    #button commands
    def Previous(self, event=None):
        '''Back to the previous step!'''
        #reload offset
        self.parent.plot_spc_quick.pack(side='left', fill='both', expand=True)
        #destroy me
        self.pack_forget()
        self.destroy()
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')


    def Finish(self, event=None):
        '''Confirm the selection in this screen'''
        #save the integral ranges
        self.trace.integral_range = (self.range_l_select, self.range_r_select)
        self.parent.temperatures.previous_spc['integral_range'] = (self.range_l_select,
                                                                  self.range_r_select)

        #finish the analysis
        self.trace.Run(broaden_width=50000)
        
        #unpack and destroy
        self.trace.analysed = True
        self.parent.plot_spc_quick.destroy()
        self.pack_forget()
        self.destroy()
        plt.close('all')
        #unbind global keys
        root.unbind('<Right>')
        root.unbind('<Left>')

        #load the overview frame
        self.parent.plot_spc_view = Frame_plot_spc_view(self.parent, self.trace)


    def Choose_ranges(self, trace):
        '''Operations and plotting for choosing spectrum integral ranges'''
        k = self.trace.phase_fit_p
        spcs = list()
        fr_list = list()
        for i, file in enumerate(trace.file_list):
            fid = FID(file, trace.file_dir)
            fid.Offset(trace.offset_range)
            fid.Shift_left(trace.mean_shl, mirroring=trace.mirroring)
            fid.Fourier()
            fid.Phase_rotate(trace.phase_fit[i])

            spcs.append(fid.spc)
            fr_list.append(fid.parameters['FR'])

        max_spc = np.unravel_index(np.argmax(spcs), (len(spcs),len(spcs[0])))[0]

        spc_fr = fid.spc_fr
        spc_mean = spcs[max_spc]
        center = int(len(spc_mean)/2)


        #plot 3
        self.axes_left1 = self.fig_left1.add_subplot(111)
        self.axes_left1.plot(np.real(spc_mean), color=colors[1], label='Re')
        self.axes_left1.plot(np.imag(spc_mean), color=colors[2], label='Im')
        self.axes_left1.axvline(x=center, color=colors[-1])
        self.axes_left1_vline_l = self.axes_left1.axvline(x=self.range_l_select,
                                                            color=colors[4])
        self.axes_left1_vline_r = self.axes_left1.axvline(x=self.range_r_select,
                                                            color=colors[4])
        self.axes_left1.set_xlim((center -200, center +200))
        self.axes_left1.set_title('Mean spectrum (Drag with left and right mouse button)')
        self.axes_left1.set_xlabel('Frequency (MHz)')
        self.axes_left1.set_ylabel('Signal (A.U.)')
        self.axes_left1.legend(loc='upper left')
        self.axes_left1.grid()

        #plot 4
        self.axes_left2 = self.fig_left2.add_subplot(111)
        for i, spc in enumerate(spcs):
            self.axes_left2.plot(np.real(spc)+np.amax(np.abs(spc_mean))*0.5*i,
                                  color=colors[i%9], label=str(i))
        self.axes_left1.axvline(x=center, color=colors[-1])
        self.axes_left2_vline_l = self.axes_left2.axvline(x=self.range_l_select,
                                                            color=colors[4])
        self.axes_left2_vline_r = self.axes_left2.axvline(x=self.range_r_select,
                                                            color=colors[4])
        self.axes_left2.set_xlim((center -200,+ center +200))
        self.axes_left2.margins(0.02, 0.02)
        self.axes_left2.set_title('All FIDs')
        self.axes_left2.set_xlabel('Frequency (MHz)')
        self.axes_left2.set_ylabel('Real part of signal (A.U.)')
        self.axes_left2.grid()

        #draggable vline event
        def Drag(event):
            '''Allows dragging of the marker in left2, recalculates mean of selected points'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.entry_range_l_var.set(int(event.xdata))
                self.range_l_select = int(event.xdata)
                #update plot
                self.axes_left1_vline_l.set_xdata(event.xdata)
                self.axes_left2_vline_l.set_xdata(event.xdata)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()

            if event.button == 3 and event.inaxes != None:
                #find the index of selected points
                self.entry_range_r_var.set(int(event.xdata))
                self.range_r_select = int(event.xdata)
                #update plot
                self.axes_left1_vline_r.set_xdata(event.xdata)
                self.axes_left2_vline_r.set_xdata(event.xdata)
                self.fig_left1.canvas.draw()
                self.fig_left2.canvas.draw()

        self.axes_left1_vline_drag = self.fig_left1.canvas.mpl_connect('motion_notify_event', Drag)
        self.axes_left2_vline_drag = self.fig_left2.canvas.mpl_connect('motion_notify_event', Drag)

        self.fig_left1.canvas.draw()
        self.fig_left2.canvas.draw()
        

class Frame_plot_spc_view(tk.Frame):
    '''Pioneer first T2 preview plot'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent)
        self.pack(side='left', anchor='n')
        
        #reference to parent
        self.parent = parent
        #reference to current trace:
        self.trace = trace
        
        #load widgets
        self.Widgets()

        #global key bind
        root.bind('<Right>', self.Confirm)

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #button commands

        def Disable(event=None):
            '''Disables and red-flags the point to avoid plotting'''
            #try:
            self.trace.disabled = not self.trace.disabled
            #except:
            #    self.trace.disabled = True
            self.Refresh_parameters()

        def Repeat(event=None):
            '''Clears the T2 trace and starts the analysis from scratch'''
            self.trace.Reinit()
            self.Confirm()
            self.trace.analysed = False

            
        #bottom button row
        self.frame_bottom = tk.Frame(self)
        self.frame_bottom.pack(side='bottom', fill='x')
        #split in columns
        self.frame_parameters = tk.Frame(self, bd=5)
        self.frame_parameters.pack(side='left', anchor='n')
        self.frame_plot = tk.Frame(self, bd=5)
        self.frame_plot.pack(side='left', anchor='n')
        self.frame_fit = tk.Frame(self, bd=5)
        self.frame_fit.pack(side='left', anchor='n', fill='y')


        #parameters
        self.label_parameters = tk.Label(self.frame_parameters,  text='Parameters')
        self.label_parameters.pack(side='top')

        self.tree_columns = ('Name','Value')
        self.tree_parameters = ttk.Treeview(self.frame_parameters, columns=self.tree_columns,
                                            show='headings', selectmode='none', height=25)
        self.tree_parameters.pack(side='top',fill='y', expand=True)

        #define column widths
        self.tree_parameters.column('Name', width=80)
        self.tree_parameters.column('Value', width=120)
        #define column names
        for column in self.tree_columns:
            self.tree_parameters.heading(column, text=column)
        #display in degrees
        #fill in params
        self.Refresh_parameters()
        
        # disable point button
        self.button_disable = ttk.Button(self.frame_parameters, text='Disable/enable Point',
                                         command=Disable, width=20)
        self.button_disable.pack(side='top')
        #redo analysis button
        self.button_repeat = ttk.Button(self.frame_parameters, text='Repeat analysis',
                                        command=Repeat, width=20)
        self.button_repeat.pack(side='top')

        #point plot
        self.fig_spc = plt.figure(dpi=100, figsize=(8,3))
        self.fig_spc.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.92)
        self.fig_spc.suptitle(self.trace.file_key, x=0.01, horizontalalignment='left')


        self.canvas_spc = FigureCanvasTkAgg(self.fig_spc, self.frame_plot)
        self.canvas_spc._tkcanvas.pack()

        #glue plot
        self.fig_glue = plt.figure(dpi=100, figsize=(8,4))
        self.fig_glue.subplots_adjust(bottom=0.15, left= 0.10, right=0.96, top=0.92)

        self.canvas_glue = FigureCanvasTkAgg(self.fig_glue, self.frame_plot)
        self.canvas_glue._tkcanvas.pack()

        #fitting range variables
        self.fit_range_l_var = tk.IntVar(self, value=0)
        self.fit_range_r_var = tk.IntVar(self, value=len(self.trace.fr_list))
        try:
            self.fit_range_l_var.set(self.trace.fit_range[0])
            self.fit_range_r_var.set(self.trace.fit_range[1])
        except:
            pass

        self.Fill_plot()
        self.Fitting_frame()
        

##        #add button to confirm selection
##        self.button_confirm = ttk.Button(self.frame_bottom, text='Confirm', command=self.Confirm)
##        self.button_confirm.pack(side='right')
##        self.button_confirm.bind('<Return>', self.Confirm)
##  
##        #add button to export spectra
##        self.button_confirm = ttk.Button(self.frame_bottom, text='Export CSV', command=self.Export)
##        self.button_confirm.pack(side='right')

    def Fitting_frame(self, event=None):
        '''Repacks/initializes the fitting frame for the selected fitting function'''

        #repack if existing:
        try:
            self.frame_fit.destroy()
        except:
            pass
        #fit frame
        self.frame_fit = tk.Frame(self, bd=5)
        self.frame_fit.pack(side='left', anchor='n', fill='y')

        #fit functions
        def Fit_lorentz(x, x0=0, a=500, g=1000):
            '''Lorentzian lineshape model'''
            return a*g/np.pi/(g**2 + (x-x0)**2)

        def Asym(x, x0, c):
            '''The smoothened function used to assymetrise'''
            b = 10
            return np.abs(c)/(1+np.exp((x-x0)*np.sign(c)*b))+1

        def Fit_lorentz_asymmetric(x, x0=0, a=500, g=1000, c=0.2):
            '''Lorentzian lineshape model with asymmetry'''
            ga = g*Asym(x, x0, c)
            aa = a*Asym(x, x0, c)
            return aa*ga/np.pi/(ga**2 + (x-x0)**2)

##        def Fit_lorentz_asymmetric(x, x0=0, a=500, g=1000, b=1, c=0):
##            '''Lorentzian with fermi changing linewidth for asymmetry'''
##            g2 = g*(1/(1+np.exp((x-x0)/b)) + c)
##            return a*g2/np.pi/(g2**2 + (x-x0)**2)

        def Fit_polynom(x, x0=0, a=1, b=0, c=0, d=0):
            '''T1 fit model for spin 3/2'''
            return a*(x-x0)**3 + b*(x-x0)**2 + c*(x-x0) + d

        def Fit_gaussian(x, x0=0, a=1, s=1000):
            '''Gaussian lineshape model'''
            return a/(s*np.sqrt(2*np.pi))*np.exp(-0.5*((x-x0)/s)**2)



        #reference to functions
        # [function, fit_params, start guess, label, tex_form]
        self.fit_names = {'Lorentz':[Fit_lorentz, ['x0', 'a', 'g'],
                                     [self.trace.fr_list[int(len(self.trace.fr_list)/2)],
                                      self.trace.spc_list_points[int(len(self.trace.fr_list)/2)],
                                      self.trace.fr_list[-1]-self.trace.fr_list[0]/2
                                      ],
                                     'a*g/pi/(g^2+(x-x0)^2)'
                                     ],
                          'Asymmetric':[Fit_lorentz_asymmetric, ['x0', 'a', 'g', 'c'],
                                        [self.trace.fr_list[int(len(self.trace.fr_list)/2)],
                                         self.trace.spc_list_points[int(len(self.trace.fr_list)/2)],
                                         self.trace.fr_list[-1]-self.trace.fr_list[0]/2,
                                         0
                                         ],
                                        'a*g(x)/pi/(g(x)^2+(x-x0)^2)'
                                        ],
                          'Polynom':[Fit_polynom, ['x0','a','b','c','d'],
                                     [self.trace.fr_list[int(len(self.trace.fr_list)/2)],
                                      1,
                                      3*self.trace.fr_list[int(len(self.trace.fr_list)/2)],
                                      0,
                                      0],
                                     'a*x^3+b*x^2+c*x+d'
                                     ],
                          'Gauss':[Fit_gaussian, ['x0', 'a', 's'],
                                     [self.trace.fr_list[int(len(self.trace.fr_list)/2)],
                                      self.trace.spc_list_points[int(len(self.trace.fr_list)/2)],
                                      self.trace.fr_list[-1]-self.trace.fr_list[0]/2
                                      ],
                                     'a*/sqrt(2*pi)*exp(-0.5*((x-x0)/s)**2)^2'
                                     ]
                          }

        def Fit():
            '''Executes the fit with given parameters and plots it'''
            Fit_function = self.fit_names[self.combo_fit_var.get()][0]
            #read values from entry boxes
            start_params = dict()
            for entry,param in zip(self.entry_params_start, param_list):
                start_params[param]=float(entry.get())
            #data points
            fit_range=(int(self.fit_range_l_var.get()), int(self.fit_range_r_var.get()))
            #print(fit_range)

            x = self.trace.fr_list
            x2 = self.trace.fr_list[slice(*fit_range)]
            y = self.trace.spc_list_points[slice(*fit_range)]

            
            #print(x,y)
            y_start = [Fit_function(xx, **start_params) for xx in x]

            #check if last parameter is enabled or not
            p_start = [start_params[key] for key in param_list]
##            if not self.check_params_start_var.get():
##                p_start.pop(-1)
##                self.entry_params_fit[-1].config(state='normal')
##                self.entry_params_fit[-1].delete(0, 'end')
##                self.entry_params_fit[-1].insert('end', 1)
##                self.entry_params_fit[-1].config(state='readonly')

            #run fit, p_optimal, p_covariance matrix
            try:
                popt,pcov = curve_fit(Fit_function, x2, y, p0=p_start)
                y_fit = [Fit_function(xx, *popt) for xx in x]
            except:
                return

            #print values to entry boxes
            for i,p in enumerate(popt):
                self.entry_params_fit[i].config(state='normal')
                self.entry_params_fit[i].delete(0, 'end')
                self.entry_params_fit[i].insert('end','%.4g' %p)
                self.entry_params_fit[i].config(state='readonly')

            #update plots
            self.axes_start_plot.set_ydata(y_start)
            self.axes_fit_plot.set_ydata(y_fit)
            self.fig_spc.canvas.draw()


            #save parameters
            self.trace.fit_params = popt
            self.trace.fit_param_cov = pcov
            self.trace.fr = popt[0]
            self.trace.width = popt[2]
            self.trace.fit_range = fit_range
##            if self.check_params_start_var.get():
##                self.trace.r = popt[-1]
##            else:
##                self.trace.r = 1

            self.Refresh_parameters()

        def Change_fit(event=None):
            '''Changes the current fitting function'''
            #update memory in parent
            self.parent.temperatures.previous_t1['fit']=self.combo_fit_var.get()
            #repack the entries
            self.Fitting_frame()
            #rerun
            #Fit()

       
        #implement more options later if necessary
        self.label_fit = tk.Label(self.frame_fit, text='Fitting function')
        self.label_fit.pack(side='top')

        self.combo_fit_var = tk.StringVar()
        try:
            self.combo_fit_var.set(self.parent.temperatures.previous_spc['fit'])
        except KeyError:
            self.combo_fit_var.set('Lorentz')
            self.parent.temperatures.previous_spc['fit']='Lorentz'
             
        self.combo_fit = ttk.Combobox(self.frame_fit, state='readonly', values=sorted(list(self.fit_names.keys())),
                                      textvar=self.combo_fit_var)
        self.combo_fit.pack(side='top')
        self.combo_fit.bind("<<ComboboxSelected>>", Change_fit)

        

        self.label_fit_fun = tk.Label(self.frame_fit, text=self.fit_names[self.combo_fit_var.get()][3], bd=5)
        self.label_fit_fun.pack(side='top')

        self.label_starting_params = tk.Label(self.frame_fit, text='Starting values', bd=5)
        self.label_starting_params.pack(side='top')

        param_list = self.fit_names[self.combo_fit_var.get()][1]
        #guesses for where params should start
        start_guess = self.fit_names[self.combo_fit_var.get()][2]
##        start_guess = [self.trace.tau_list[self.trace.mean_range[0]-5],
##                       np.mean(self.trace.area_list[slice(*self.trace.mean_range)]),
##                       -self.trace.area_list[0]/self.trace.area_list[-1],
##                       1]
##        if self.combo_fit_var.get() == 'Spin 3/2':
##            start_guess[0] = start_guess[0]*6

        #start parameters entry rows
        self.frame_params_start = list()
        self.label_params_start = list()
        self.entry_params_start = list()
        for i,param in enumerate(param_list):
            self.frame_params_start.append(tk.Frame(self.frame_fit))
            self.frame_params_start[i].pack(side='top', fill='y')

            self.label_params_start.append(tk.Label(self.frame_params_start[i], text=param+' = '))
            self.label_params_start[i].pack(side='left', anchor='e')

            self.entry_params_start.append(tk.Entry(self.frame_params_start[i],
                                                    width=10, justify='right'))
            self.entry_params_start[i].insert(0, '%.4g' % start_guess[i])
            self.entry_params_start[i].pack(side='left', anchor='e')

        #check button for stretch
##        self.check_params_start_var = tk.BooleanVar(self, 0)
##        try:
##            if self.trace.r != 1:
##                self.check_params_start_var.set(1)
##        except AttributeError: pass
##        self.check_params_start = (ttk.Checkbutton(self.frame_params_start[-1],
##                                                   variable=self.check_params_start_var))
##        self.check_params_start.pack(side='left')
            
        self.button_fit = ttk.Button(self.frame_fit, text='Retry fit', command=Fit)
        self.button_fit.pack(side='top')

        self.label_fit_params = tk.Label(self.frame_fit, text='Fitted values', bd=5)
        self.label_fit_params.pack(side='top')

        #fit results entry rows
        self.frame_params_fit = list()
        self.label_params_fit = list()
        self.entry_params_fit = list()
        for i,param in enumerate(param_list):
            self.frame_params_fit.append(tk.Frame(self.frame_fit))
            self.frame_params_fit[i].pack(side='top', fill='y')

            self.label_params_fit.append(tk.Label(self.frame_params_fit[i], text=param+' = '))
            self.label_params_fit[i].pack(side='left')

            self.entry_params_fit.append(tk.Entry(self.frame_params_fit[i], width=10,
                                                  state='readonly', justify='right'))
            self.entry_params_fit[i].pack(side='left')

        #run first lap of fit
        Fit()

        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_fit, text='Confirm', command=self.Confirm)
        self.button_confirm.pack(side='bottom')
        self.button_confirm.bind('<Return>', self.Confirm)

        #add export csv button
        self.button_export = ttk.Button(self.frame_fit, text='Export CSV', command=self.Export)
        self.button_export.pack(side='bottom')
        self.button_export.bind('<F5>', self.Export)

        #add export glue button
        self.button_export_glue = ttk.Button(self.frame_fit, text='Export Glue', command=self.Export_glue)
        self.button_export_glue.pack(side='bottom')
        self.button_export_glue.bind('<F6>', self.Export_glue)

    def Confirm(self, event=None):
        '''Confirm the selection in this screen'''
        #unpack, dont destroy untill series is done, in case corrections are needed
        self.parent.temperatures.wait.set(False)
        self.pack_forget()
        self.destroy()
        plt.close('all')

        #move to later stages
        self.trace.analysed = True

    def Refresh_parameters(self):
        '''refreshes the parameters table'''
        self.tree_parameters.delete(*self.tree_parameters.get_children())
        self.trace.Get_params()
        for item in GLOBAL_spc_displayed_params:
            try:
                pair = (item, self.trace.__dict__[item])
                self.tree_parameters.insert('', 'end', values=pair)
            except: pass

    def Export(self):
        '''Saves the plotted data into a CSV file for further analysis'''
        file_name = self.trace.__dict__['file_key'] + '.csv'
        file_directory = os.path.join('data', self.parent.current_experiment, 'csv', 'spc')

        #make the csv folder for old experiments
        try:
            os.mkdir(file_directory)
        except: pass

        #write file
        with open(os.path.join(file_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            #name row
            writer.writerow(['fr(MHz)', 'area(A.U.)'])
            #data
            for i in range(len(self.trace.fr_list)):
                row = [self.trace.fr_list[i], self.trace.spc_list_points[i]]
                writer.writerow(row)

        tk.messagebox.showinfo('Export complete', 'The file was saved as '+file_name)

    def Export_glue(self):
        '''Saves the glue points into a CSV file'''
        file_name = self.trace.__dict__['file_key'] + '.csv'
        file_directory = os.path.join('data', self.parent.current_experiment, 'csv', 'glue_spc')

        #make the csv folder for old experiments
        try:
            os.mkdir(file_directory)
        except: pass

        #define data lines
        x = self.trace.spc_fr
        y = self.trace.spc_sig_real
        y2 = self.trace.spc_sig_imag
        y3 = np.sqrt(y**2 + y2**2)

        #write file
        with open(os.path.join(file_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            #name row
            writer.writerow(['fr(MHz)', 'Re(A.U.)','Im(A.U.)','Abs(A.U.)'])
            #data
            for i in range(len(x)):
                row = [x[i], y[i], y2[i], y3[i]]
                writer.writerow(row)

        tk.messagebox.showinfo('Export complete', 'The file was saved as '+file_name)

    def Fill_plot(self):
        '''Plots the T1 trend and fits it'''
        #data lines
        x = self.trace.fr_list
        y = self.trace.spc_list_points

##        sorting = np.argsort(x)
##        x = np.array(x)[sorting]
##        y = np.array(y)[sorting]


        #point plot
        self.axes = self.fig_spc.add_subplot(111)
        self.axes.plot(x, y, marker='o', linestyle='-', color=colors[1], label='Data')
        self.axes_start_plot, = self.axes.plot(x, y, color=colors[3],
                                               linestyle='dashed', label='Fit start')
        self.axes_fit_plot, = self.axes.plot(x, y, color=colors[4], label='Fit')
        self.axes_vline_l = self.axes.axvline(x=self.trace.fr_list[self.fit_range_l_var.get()], color=colors[4])
        self.axes_vline_r = self.axes.axvline(x=self.trace.fr_list[self.fit_range_r_var.get()-1], color=colors[4])
        self.axes.set_title('Delta spectrum')
        self.axes.set_xlabel(r'$\nu$ (MHz)')
        self.axes.set_ylabel('Area (A.U.)')
        legend = self.axes.legend(loc='lower right')
        self.axes.grid()


        x = self.trace.spc_fr
        y = self.trace.spc_sig_real
        y2 = self.trace.spc_sig_imag
        y3 = np.sqrt(y**2 + y2**2)
        #glue plot
        self.axes_glue = self.fig_glue.add_subplot(111)
        self.axes_glue.plot(x, y, color=colors[1], label='Re')
        self.axes_glue.plot(x, y2, color=colors[2], label='Im')
        self.axes_glue.plot(x, y3, color=colors[3], label='Abs')
        self.axes_glue.set_title('Glued spectrum')
        self.axes_glue.set_xlabel(r'$\nu$ (MHz)')
        self.axes_glue.set_ylabel('Signal (A.U.)')
        legend = self.axes_glue.legend(loc='lower right')
        self.axes_glue.grid()

        #draggable vline event
                                            
        def Drag(event):
            '''Allows dragging of the markers for fit range on spectrum plot'''
            if event.button == 1 and event.inaxes != None:
                #find the index of selected points
                self.fit_range_l_var.set(np.searchsorted(self.trace.fr_list, event.xdata, side='right'))
                #print(self.fit_range_l_var.get())
                #self.range_l_select = int(event.xdata)
                #update plot
                self.axes_vline_l.set_xdata(event.xdata)
                self.fig_spc.canvas.draw()

            if event.button == 3 and event.inaxes != None:
                #find the index of selected points
                self.fit_range_r_var.set(np.searchsorted(self.trace.fr_list, event.xdata, side='left'))
                #print(self.fit_range_r_var.get())
                #self.range_r_select = int(event.xdata)
                #update plot
                self.axes_vline_r.set_xdata(event.xdata)
                self.fig_spc.canvas.draw()

        self.axes_vline_drag = self.fig_spc.canvas.mpl_connect('motion_notify_event', Drag)
        
        
class Frame_plot_spc_frvt(tk.Frame):
    '''T1vT trend plotting'''
    def __init__(self, parent, trace):
        '''makes the subframe and fills it up'''
        tk.Frame.__init__(self, parent)
        self.pack(side='left', anchor='n')
        
        #reference to parent
        self.parent = parent
        #reference to current series
        self.trace = trace

        #counter for plots
        self.counter = 0
        
        #load widgets
        self.Widgets()

    def Widgets(self):
        '''Builds all the subframes and canvases'''
        #button commands
        def Confirm(event=None):
            '''Confirm the selection in this screen'''
            #unpack, dont destroy untill series is done, in case corrections are needed
            self.parent.temperatures.wait.set(False)
            self.pack_forget()
            self.destroy()
            plt.close('all')

            self.parent.traces.button_show.config(state='normal')

        #split in columns
        self.frame_plot_left = tk.Frame(self)
        self.frame_plot_left.pack(side='left', anchor='n')
        self.frame_plot_right = tk.Frame(self)
        self.frame_plot_right.pack(side='left', anchor='n')

        #plot frames
        self.frame_plot1 = tk.Frame(self.frame_plot_left, bd=5)
        self.frame_plot1.pack(side='top', anchor='n')
        self.frame_plot2 = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_plot2.pack(side='top', anchor='n')
        self.frame_plot3 = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_plot3.pack(side='top', anchor='n')

        #buttons frame
        self.frame_buttons = tk.Frame(self.frame_plot_right, bd=5)
        self.frame_buttons.pack(side='top', anchor='e')
        
        #T1 plot
        self.fig_t1vt = plt.figure(dpi=100, figsize=(7,4.5))
        self.fig_t1vt.subplots_adjust(bottom=0.12, left= 0.11, right=0.96, top=0.94)
        self.canvas_t1vt = FigureCanvasTkAgg(self.fig_t1vt, self.frame_plot1)
        self.canvas_t1vt._tkcanvas.pack()

        #fr plot
        self.fig_fr = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_fr.subplots_adjust(bottom=0.18, left= 0.11, right=0.96, top=0.90)
        self.canvas_fr = FigureCanvasTkAgg(self.fig_fr, self.frame_plot2)
        self.canvas_fr._tkcanvas.pack()

        #stretch plot
        self.fig_r = plt.figure(dpi=100, figsize=(7,2.5))
        self.fig_r.subplots_adjust(bottom=0.18, left= 0.11, right=0.96, top=0.90)
        self.canvas_r = FigureCanvasTkAgg(self.fig_r, self.frame_plot3)
        self.canvas_r._tkcanvas.pack()


        #add button to confirm selection
        self.button_confirm = ttk.Button(self.frame_buttons, text='Confirm', command=Confirm)
        self.button_confirm.pack(side='right')
        self.button_confirm.bind('<Return>', Confirm)

        #plot the stuff
        self.Fill_plot()

        #add button to export parameters
        self.button_export = ttk.Button(self.frame_buttons, text='Export CSV', command=self.Export)
        self.button_export.pack(side='right')



    def Add_trace(self, trace):
        '''Adds traces to plots using given trace'''
        #initialize lists
        x = list()
        fr = list()
        width = list()
        amp = list()
        for temp in self.trace:
            if self.trace[temp].analysed and not self.trace[temp].disabled:

                x.append(self.trace[temp].temp_set)
                #maby calculate some center frequency at some point?
                fr.append(self.trace[temp].fr)
                width.append(self.trace[temp].width)
                amp.append(self.trace[temp].fit_params[1])
                name = self.trace[temp].file_key
        #sort by temperature
        sorting = np.argsort(x)
        x = np.array(x)[sorting]
        fr = np.array(fr)[sorting]
        width = np.array(width)[sorting]
        amp = np.array(amp)[sorting]
                           

        

        #draw trace
        self.axes_1.plot(x, fr, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')
        self.axes_2.plot(x, fr, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')
        self.axes_3.plot(x, 2*width, 'bo', color=colors[self.counter],
                         label=self.parent.current_trace, linestyle='dashed')

        #save for export
        self.data = dict()
        self.data['T'] = x
        self.data['fr'] = fr
        self.data['FWHM'] = 2*width
        self.data['amp'] = amp/np.pi/width
  

        #increase plot counter
        self.counter += 1

    def Export(self):
        '''Saves the plotted data into a CSV file for further analysis'''
        file_name = self.parent.current_trace + '.csv'
        file_directory = os.path.join('data', self.parent.current_experiment, 'csv', 'spc')

        #make the csv folder for old experiments
        try:
            os.mkdir(file_directory)
        except: pass

        #write file
        with open(os.path.join(file_directory, file_name), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            #name row
            writer.writerow(['T(K)', 'fr(MHz)','FWHM(MHz)','amplitude'])
            #data
            for i in range(len(self.data['T'])):
                row = [self.data['T'][i], self.data['fr'][i], self.data['FWHM'][i], self.data['amp'][i]]
                writer.writerow(row)

        tk.messagebox.showinfo('Export complete', 'The file was saved as '+file_name)

    def Fill_plot(self):
        '''Creates the plots for T1vT'''

        self.axes_1 = self.fig_t1vt.add_subplot(111)
        self.axes_1.set_xscale('log')
        #self.axes_1.set_yscale('log')
        self.axes_1.set_title('Frequency temperature dependence')
        self.axes_1.set_xlabel('Temperature (K)')
        self.axes_1.set_ylabel(r'Frequency (MHz)')
        #self.axes_1.legend(loc='lower right')
        self.axes_1.grid()

        self.axes_2 = self.fig_fr.add_subplot(111)
        self.axes_2.set_title('Frequency temperature dependence')
        self.axes_2.set_xlabel('Temperature (K)')
        self.axes_2.set_ylabel('Frequency (MHz)')
        #self.axes_2.get_yaxis().get_major_formatter().set_useOffset(False)
        self.axes_2.margins(0.05, 0.1)
        self.axes_2.grid()

        self.axes_3 = self.fig_r.add_subplot(111)
        self.axes_3.set_title('Linewidth')
        self.axes_3.set_xlabel('Temperature (K)')
        self.axes_3.set_ylabel('Linewidth')
        #self.axes_3.get_yaxis().get_major_formatter().set_useOffset(False)
        self.axes_3.margins(0.05, 0.1)
        self.axes_3.grid()


   
class Main_application(tk.Frame):
    '''Main application calling all the sub sections'''
    def __init__(self, parent, *args, **kwargs):
        '''Initializes the main application as a frame in tkinter'''
        #check for computers screen resolution
        width_px = root.winfo_screenwidth()
        height_px = root.winfo_screenheight()
        if width_px < 1720:
            self.Warn_resolution()
        
        tk.Frame.__init__(self, parent, height=1020, width =1720, *args, **kwargs)
        self.parent = parent
        # sets the window title
        self.parent.wm_title('NMR data analysis and overview')
        #sets the window minimal size
        self.parent.minsize(width=1880, height=770)
        # allow editing the exit command
        self.parent.protocol('WM_DELETE_WINDOW', self.On_close)
        #makes the window strechable
        self.pack(fill='both', expand=True)
        self.pack()

        #place to save Experiment_data classes
        self.Open_data()
        
        #calls subframes and packs them
        self.Sub_frames()

    def Open_data(self):
        '''Opens all experiments from folders data subdirectory'''
        self.data = dict()

        #adds the existing experiments
        if os.path.isdir('data'):
            for entry in os.scandir('data'):
                if entry.is_dir():  
                    #initiates a dict of the experiment data classes
                    self.data[entry.name] = Experiment_data(entry.name)
        #makes the raw experiment folder
        else:
            msg = 'The current directory does not contain the correct file structure.' \
                  '\nCreate new folders in current directory?'
            if tk.messagebox.askyesno('No data in current directory', msg):
                os.mkdir('data')
        

    def Sub_frames(self):
        '''Creates all the subframes and positions them'''
        #first column
        self.experiments = Frame_experiments(self)
        #self.experiments.pack(side='left', fill='y')

        #second column
        self.traces = Frame_traces(self)
        #self.traces.pack(side='left', fill='y')

        #3rd column
        self.temperatures = Frame_temperatures(self)
        #self.temperatures.pack(side='left', fill ='y')

        ## construct it every time instead (to avoid memory problems)
        #4th column, plotter
        class Tracer():
            pass
        #self.plot1 = Frame_plot_T1_view(self, Tracer)
        #temporary pack
        #self.plot1.pack(side='left', fill='both', expand=True)

    def On_close(self):
        '''Actions to execute when the master window is closed with ('X')'''
        msg = 'Are you sure you want to close the program?\n' \
              'Unsaved data will be lost!'
        if tk.messagebox.askokcancel('Quit', msg):
            #close all plots and figures
            plt.close('all')
            self.parent.destroy()

    def Warn_resolution(self):
        '''Warns the user that the resolution of the program is not optimal for the monitor'''
        msg = 'The resolution of the program is larger than the resolution of the monitor!\n' \
              'The program might not function or appear properly!'
        tk.messagebox.showerror('Error', msg)
        #implement parameters for smaller figure sizes to account for smaller resolution!!
        
def Error_incomplete():
    '''Lets user know that the content doesnt exist yet'''
    tk.messagebox.showerror('Error', 'The function is not yet implemented!')
        
def Quit_program():
    '''destroys root and quits the tkinter'''
    root.destroy()
    
if __name__ == '__main__':
    '''Initializes tkinter and main application if this is a standalone file'''
    root = tk.Tk()   
    Main_application(root)
    root.mainloop()


