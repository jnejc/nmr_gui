#######################################
# program for analyzing nmr data (from 7NMR)
# created by Nejc Jansa
# nejc.jansa@ijs.si
# document creation 14.11.2016
# last version: 06.11.2018
#######################################

#conventions:
#functions capitalized, variables small
#data lists should be numpy arrays np.array(), once they get big
#ranges should go around as tuples:
#last element: tup=(-5, None), then make a[slice(*tup)]
#never put random integers in function, always name them and default call them


#points to consider:
#

#(minimal) necessary imports

# Versions of packages:
# numpy-1.11.2+mkl-cp35-cp35m-win_amd64.whl
# scipy-0.18.1-cp35-cp35m-win_amd64.whl
# matplotlib-2.0.0-cp35-cp35m-win_amd64.whl

#should minimalize...?
import os #os path functions for importing
import re #searching strings,etc (regex)
import matplotlib           #plots
import numpy as np          #numpy for all numerical
import matplotlib.pyplot as plt #short notation for plots
from scipy.optimize import curve_fit    #fitting algorithm from scipy
import pickle               #pickle for saving python objects
#from matplotlib.gridspec import GridSpec       #widgets for matplotlib plots
#from matplotlib.widgets import SpanSelector
#from matplotlib.widgets import Button

#plot colors from colorbrewer
colors = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf','#999999']

#set some global settings for plots
plot_font = {'family': 'Calibri', 'size': '12'}
matplotlib.rc('font', **plot_font)  # make the font settings global for all plots



### global variables/lists (everything that might cahnge at some point and should be acessible from multiple points)
GLOBAL_pkl_list = ['series','raw_file_list','raw_dir','possible_series']


#create a FID class that takes care of extracting and analyzing the fid files from 7nmr
class FID:
    '''Class that opens and analyzes data from the 7nmr files'''

    def __init__(self, file_name, file_dir):
        '''Initializes the class taking the name and directory of the file'''
        self.file_name = file_name
        
        self.Open_file(file_name, file_dir)
        
    ### routines        
        
    def Open_file(self, file_name, file_dir):
        '''Opens the file and extracts parameters as dict and the lists of data'''
        self.parameters = dict()

        with open(os.path.join(file_dir, file_name)) as f:
            lines = list(f)
            #remove the \n
            lines = [l.strip() for l in lines]

            #extract the parameters (old rules list) 
            # initialize a dictionary [item : value , item2 : value2, ...]   
            
            for l in lines[lines.index("[PARAMETERS]")+1 : lines.index("[DATA]")]:
                tmp = l.split("=")
                #skip the non a=b lines
                if len(tmp) != 2: continue
                #convert what you can into floats
                try:
                    self.parameters[tmp[0]] = float(tmp[1])
                #when error occurs leave it as string
                except:
                    self.parameters[tmp[0]] = tmp[1]

            #extract the data columns 
            self.x1 = []
            self.x2 = []
            for l in lines[lines.index("[DATA]")+1 : ]:
                #tmp=re.findall(exp_notation, l)
                tmp = l.split()
                self.x1.append(float(tmp[0]))
                self.x2.append(float(tmp[1]))

        #make a complex numpy array for editing
        self.x = np.array(self.x1) + 1j * np.array(self.x2)
        self.x_len = len(self.x)

    def Run(self, broaden_width=None, shl=None, mirroring=None, zero_fill=1):
        '''Runs the basic analysis on self'''
        self.Offset(offset_range = (-200,-1))
        self.Find_SHL(3)
        if shl:
            self.Shift_left(shl)
        else:
            self.Shift_left(self.shl, mirroring=mirroring)
        if broaden_width:
            self.Line_broaden(broaden_width)
        self.Fourier(zero_fill=zero_fill)
        self.Phase_spc()
        self.Phase_rotate(self.phase_spc)
        self.Plot_fid()
        self.Plot_spc()

    def Offset(self, offset_range=(-200,-1)):
        '''Calculates mean offset of last points of x and sets it to 0'''
        self.mean = np.mean(self.x[slice(*offset_range)])
        self.x = self.x - self.mean
        self.offset_range = offset_range

    def Find_SHL(self, convolution=1):
        '''Smooths the list with symmetric convolution and finds maximal element'''
        convolved = np.convolve(np.ones(2*convolution-1), self.x, 'same')
        self.shl = np.argmax(np.absolute(convolved))

    def Shift_left(self, shl, mirroring=None):
        '''Shifts the array to left and pads with zeroes or mirrors to back'''
        if not mirroring:
            self.x = np.delete(self.x, np.s_[0:shl])
            self.x = np.append(self.x, np.zeros(shl))
            self.offset_range = (self.offset_range[0] - shl, None)
            self.mirroring=False
        else:
            self.x = np.roll(self.x, -shl)
            self.offset_range = (self.offset_range[0] - shl, None)
            self.mirroring=True

    def Line_broaden(self, broaden_width):
        '''Broadens the spectral lines by putting a gaussian envelope over the fid'''
        dw = self.parameters['DW']
        if not self.mirroring:
            a = np.array(range(self.x_len))
        else:
            #adds the negative side of the broadening gaussian on the second half
            a = np.concatenate((range(self.x_len//2), range(-self.x_len//2, 0)))

        self.broaden_width = broaden_width
        self.x = np.multiply(self.x, np.exp(-(a*broaden_width*dw)**2/(4*np.log(2))))

    def Fourier(self, zero_fill=1):
        '''Fourier transforms the fid and sorts the lists b frequency'''
        fill_len = self.x_len * 2**zero_fill
        time_step = self.parameters['DW'] #s
        center_frequency = self.parameters['FR'] #MHz
        #make fft on data
        if not self.mirroring:
            #regular fill
            spectrum = np.fft.fft(self.x, fill_len)
        else:
            #fill in center
            spectrum = np.fft.fft(np.insert(self.x, int(self.x_len/2), np.zeros(int(fill_len/2))), fill_len)
        #generate correct list of frequencies for the fft
        frequencies = -np.fft.fftfreq(fill_len, time_step)/10**6 + center_frequency #MHz
        #sorting and saving
        sort_list = frequencies.argsort()
        self.spc = spectrum[sort_list]
        self.spc_fr = frequencies[sort_list]
        self.spc_len = fill_len

    def Phase_fid(self, phase_range=(0,None)):
        '''Calculates the phase of the signal form the fid (in given range)'''
        self.phase_fid = np.angle(np.sum(self.x[slice(*phase_range)]))

    def Phase_spc(self, phase_range=(0,None)):
        '''Calculates the phase of the signal from the spc (in given range)'''
        self.phase_spc = np.angle(np.sum(self.spc[slice(*phase_range)]))

    def Phase_rotate(self, phase):
        '''Rotates the phase of the fid and spc by the given phase'''
        self.x = self.x * np.exp(-1j * phase)
        if 'spc' in self.__dict__:
            self.spc = self.spc * np.exp(-1j * phase)

    def Integral_spc(self, integral_range):
        '''Integrates the area under the real part of the spectrum in given (list index) range'''
        self.area_spc = np.sum(self.spc.real[slice(*integral_range)])
        self.integral_range_spc = integral_range

    def Integral_fid(self, integral_range):
        '''Integrates the area under the absolute value of fid in given (list index) range'''
        self.area_fid = np.sum(self.spc.real[slice(*integral_range)])

    def Plot_fid(self, x_range=None, time_axis=False):
        '''Shows a plot of the current fid'''
        plt.figure()
        #add the traces
        if time_axis:
            self.t = np.linspace(0, self.x_len-1, self.x_len) * self.parameters['DW']*1.0e+6
            plt.plot(self.t, self.x.real, color=colors[1],label="Re")
            plt.plot(self.t, self.x.imag, color=colors[2],label="Im")
            plt.plot(self.t, np.absolute(self.x), color=colors[3],label="Abs")
            plt.xlabel("t (us)")
        else:
            plt.plot(self.x.real, color=colors[1],label="Re")
            plt.plot(self.x.imag, color=colors[2],label="Im")
            plt.plot(np.absolute(self.x), color=colors[3],label="Abs")
            plt.xlabel("t (index)")
            
        try:
            off= self.x_len + self.offset_range
            plt.axvline(x=off,color=colors[-1])
        except: pass
        #plot labels and frames
        plt.title("Current fid function")
        plt.ylabel("signal")
        plt.grid()
        plt.legend(loc='upper right')
        #plot range
        if x_range:
            plt.xlim(xmax=x_range)
        #post plot
        plt.show()

    def Plot_spc(self, plot_range=None, frequency_axis=True):
        #plots the current fid
        plt.figure()
        #add the traces
        if frequency_axis:
            plt.plot(self.spc_fr, self.spc.real, color=colors[1],label="Re",marker='.')
            plt.plot(self.spc_fr, self.spc.imag, color=colors[2],label="Im",marker='.')
            plt.axvline(x=self.parameters['FR'],color=colors[-1])
            #plot range
            if plot_range:
                plt.xlim((self.parameters['FR']-plot_range,self.parameters['FR']+plot_range))
            else:
                plt.xlim((self.parameters['FR']-0.5,self.parameters['FR'] +0.5))
        else:
            plt.plot(self.spc.real, color=colors[1],label="Re",marker='.')
            plt.plot(self.spc.imag, color=colors[2],label="Im",marker='.')
        try:
            l= self.spc_fr[self.integral_range_spc[0]]
            r= self.spc_fr[self.integral_range_spc[1]]
            plt.axvline(x=l,color=colors[-1])
            plt.axvline(x=r,color=colors[-1])
        except: pass
        #plot labels and frames
        plt.title("Current spc function")
        plt.xlabel("t (index)")
        plt.ylabel("signal")
        plt.grid()
        plt.legend(loc='upper right')

        #print the plot
        plt.show()

class Glue_spc:
    '''takes a series of fid measurements and makes a wide spectrum'''
    # in future give it all the properties of a FID class?

    def __init__(self, file_key, file_dir):
        '''Initializes the class and sets the file keys and directory and makes the list'''
        self.file_key = file_key
        self.file_dir = file_dir
        self.Find_files()

        self.analysed = False
        self.disabled = False
        self.mirroring = False

        self.Get_params()

    def Reinit(self):
        '''deletes all content and reinitializes class'''
        file_key = self.file_key
        file_dir = self.file_dir
        #clear parameters and restart
        self.__dict__.clear()
        self.__init__(file_key, file_dir)

    def Run2(self, offset_range=(-200,None), shl=None, fr_density=1000, broaden_width=None, fit=True, mirroring=None):
        '''Executes the analysis functions'''
        self.Get_shl(shl=shl, offset_range=offset_range)
        self.Get_phase()
        self.Get_spc(broaden_width=broaden_width, fit=fit, fr_density=fr_density, mirroring=mirroring)
        self.Plot_joined_spc()

    def Run(self, broaden_width=None, fr_density=10):
        '''finnishes the analysis for gui'''
        #glued spc:
        #start the glued spectrum array
        self.spc_fr =  np.linspace(self.fr_min - self.fr_step, self.fr_max + self.fr_step,
                                   len(self.file_list)*fr_density)
        self.spc_sig_real = np.zeros(len(self.spc_fr))
        self.spc_sig_imag = np.zeros(len(self.spc_fr))

        #list of individual spectra
        self.spc_list_real = list()
        self.spc_list_imag = list()
        self.spc_list_points = list()
        self.fr_list = list()

        for f, phase in zip(self.file_list, self.phase_fit):
            fid = FID(f, self.file_dir)
            fid.Offset(self.offset_range)
            fid.Shift_left(self.mean_shl, mirroring=self.mirroring)
            if broaden_width:
                fid.Line_broaden(broaden_width)
            fid.Fourier()
            fid.Phase_rotate(phase)
            #integral for point spc
            fid.Integral_spc(self.integral_range)
            #add data
            self.spc_list_points.append(fid.area_spc)
            self.fr_list.append(fid.parameters['FR'])
            #interpolate the spectrum to new frequencies
            interp_real = np.interp(self.spc_fr, fid.spc_fr, fid.spc.real)
            interp_imag = np.interp(self.spc_fr, fid.spc_fr, fid.spc.imag)
            self.spc_list_real.append(interp_real)
            self.spc_list_imag.append(interp_imag)
            #accumulate total spectrum
            self.spc_sig_real += interp_real
            self.spc_sig_imag += interp_imag

        self.broaden_width = broaden_width
        self.fr_density = fr_density


    def Find_files(self):
        '''Makes a list of all files with key and 000-999.DAT'''
        dir_list = os.listdir(self.file_dir)
        key = '^' + self.file_key + '-[0-9]*.DAT$'
        #save the sorted list
        def Sort_key(item):
            '''sort key to sort files by last number xxx.DAT'''
            return int(item.split('-')[-1][:-4])
        self.file_list = sorted([i for i in dir_list if re.search(key, i)], key=Sort_key)

    def Get_params(self):
        '''Extracts useful constant parameters for display from FIDs and saves into trace'''

        #initiate a representable FID
        fid = FID(self.file_list[-1], self.file_dir)
        #copy other usefull values
        self.TAU = str(1000000*fid.parameters['TAU'])+'u'
        self.D1 = str(1000000*fid.parameters['D1'])+'u'
        self.D3 = str(1000000*fid.parameters['D3'])+'u'
        self.D9 = str(int(1000*fid.parameters['D9']))+'m'
        self.NS = int(fid.parameters['NS'])


    def Get_shl(self, offset_range=(-200,None), shl=None):
        '''Checks the suggested SHL and saves the frequency list'''
        self.shl_list = list()
        self.fr_list = list()
        for f in self.file_list:
            fid = FID(f, self.file_dir)
            fid.Offset(offset_range)
            fid.Find_SHL()
            self.shl_list.append(fid.shl)
            self.fr_list.append(fid.parameters['FR'])

        #set the frequency range
        self.fr_min, self.fr_max = (min(self.fr_list), max(self.fr_list))
        self.fr_step = (self.fr_max - self.fr_min)/len(self.file_list)

        self.offset_range=offset_range
        
        print(self.shl_list)
        #choose offset by hand :(
        if not shl:
            self.shl = int(input('Choose the shl value: '))

    def Get_phase(self, shl=None, offset_range=None, fit_range=None):
        '''Gets the phases from spectra'''
        self.phase_list = list()

        if shl:
            self.shl = shl
        if offset_range:
            self.offset_range = offset_range
        if fit_range:
            self.fit_range = fit_range

        for f in self.file_list:
            fid = FID(f, self.file_dir)
            fid.Offset(self.offset_range)
            fid.Shift_left(self.shl)
            fid.Fourier()
            fid.Phase_spc()
            self.phase_list.append(fid.phase_spc)

        #linear fit to phases
        def Lin_fit(x, k=1, n=0):
            return k*x + n
        #axes
        x=self.fr_list
        y=np.unwrap(self.phase_list, 0.5*np.pi)

        #cange fit ranges
        if fit_range:
            x2=x[slice(*fit_range)]
            y=y[slice(*fit_range)]
        
        #starting values
        p_start=[1,0]
        #run fit
        popt,pcov = curve_fit(Lin_fit, x2, y, p0=p_start)
        self.phase_fit = [Lin_fit(xx, *popt) for xx in x]
        #save fit params
        self.phase_fit_p = popt
        
    def Get_spc(self, fit=True, fr_density=100, broaden_width=None, mirroring=None):
        '''Gets the individual spectra and the total spectrum'''
        #start the glued spectrum array
        self.spc_fr =  np.linspace(self.fr_min - self.fr_step, self.fr_max + self.fr_step,
                                   len(self.file_list)*fr_density)
        self.spc_sig_real = np.zeros(len(self.spc_fr))
        self.spc_sig_imag = np.zeros(len(self.spc_fr))

        #list of individual spectra
        self.spc_list_real = list()
        self.spc_list_imag = list()

        for f, phase in zip(self.file_list, self.phase_fit):
            fid = FID(f, self.file_dir)
            fid.Offset(self.offset_range)
            fid.Shift_left(self.shl, mirroring=mirroring)
            if broaden_width:
                fid.Line_broaden(broaden_width)
            fid.Fourier()
            if fit:
                fid.Phase_rotate(phase)
            else:
                fid.Phase_spc()
                fid.Phase_rotate(fid.phase_spc)
            #interpolate the spectrum to new frequencies
            interp_real = np.interp(self.spc_fr, fid.spc_fr, fid.spc.real)
            interp_imag = np.interp(self.spc_fr, fid.spc_fr, fid.spc.imag)
            self.spc_list_real.append(interp_real)
            self.spc_list_imag.append(interp_imag)
            #accumulate total spectrum
            self.spc_sig_real += interp_real
            self.spc_sig_imag += interp_imag


    def Plot_joined_spc(self):
        '''test function for plotting the joined spectra'''
        plt.figure(dpi=100,figsize=(12,6))
        #add the traces
        for i in range(len(self.file_list)):
            plt.plot(self.spc_fr, self.spc_list_real[i], color=colors[i%8], label=str(self.fr_list[i])+'MHz')
            plt.axvline(x=self.fr_list[i] , color=colors[i%8])
        plt.plot(self.spc_fr, self.spc_sig_real, color=colors[-1],label="Re",marker='.')
        #plot labels and frames
        plt.title("Joined spectrum")
        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Summed signal")
        plt.grid()
        #plt.legend(loc='upper right')
        plt.xlim(self.spc_fr[0], self.spc_fr[-1])
        #print the plot
        plt.show()

    def Plot_phase(self):
        '''Show how the phase is selected along the measurement'''
        #what to do about the phase jumps?
        plt.figure()
        #add the traces
        plt.plot(self.fr_list, np.unwrap(self.phase_list), color=colors[1],label="phase",marker='.')
        try:
            plt.plot(self.fr_list, self.phase_fit, color=colors[2],label="lin fit",marker='.')
        except: pass
        #plot labels and frames
        plt.title("Current spc function")
        plt.xlabel("t (index)")
        plt.ylabel("signal")
        plt.grid()
        plt.legend(loc='upper left')
        #print the plot
        plt.show()

    def Plot_fid_all(self, height_fact=0.5):
        '''Plots all FIDs'''
        fids = list()
        for file in self.file_list:
            fid = FID(file, self.file_dir)
            fids.append(fid.x)

        height = np.amax(np.abs(fids))
        
        plt.figure(figsize=(8,6))
        for i,fid in enumerate(fids):
            plt.plot(np.abs(fid) + height*height_fact*i, color=colors[i % len(colors)], label=str(i))
        plt.title("Fids")
        plt.xlim(0,len(fids[0]))
        plt.xlabel("t (index)")
        plt.ylabel("signal")
        plt.grid()
        #plt.legend(loc='upper left')
        #print the plot
        plt.show()

    def Plot_spc_all(self, broaden_width=None, fit=False):
        '''Plots all spectra'''
        spcs = list()
        for f, phase in zip(self.file_list, self.phase_fit):
            fid = FID(f, self.file_dir)
            fid.Offset(self.offset_range)
            fid.Shift_left(self.shl)
            if broaden_width:
                fid.Line_broaden(broaden_width)
            fid.Fourier()
            if fit:
                fid.Phase_rotate(phase)
            else:
                fid.Phase_spc()
                fid.Phase_rotate(fid.phase_spc)

        height = np.amax(np.abs(spcs[-1]))
        
        plt.figure(figsize=(8,6))
        for i,spc in enumerate(spcs):
            plt.plot(np.abs(fid)+ height*0.5*i, color=colors[i % len(colors)], label=str(i))
        plt.title("Fids")
        plt.xlabel("t (index)")
        plt.ylabel("signal")
        plt.grid()
        plt.legend(loc='upper left')
        #print the plot
        plt.show()
        
    def Quick_spc(self, offset_range=(-200,-1), phase_range=(0,-1), shl_convol=1, integral_range=None):
        '''Runs through all files to determine shls values and preplot the fids'''
  
        self.quick_shl = list()
        self.temp_list = list()
        self.temp_list2 = list()
        self.fr_list = list()

        fid = FID(self.file_list[-1], self.file_dir)
        self.temp_set = fid.parameters.get('_ITC_R0',0)

        #integral range hevristics
        if not integral_range:
            integral_range = (int(fid.parameters['D3']/fid.parameters['DW']/2),
                              int(fid.parameters['D3']/fid.parameters['DW']*2))

        for file in self.file_list:
            fid = FID(file, self.file_dir)
            fid.Offset(offset_range)
            fid.Find_SHL(convolution=shl_convol)

            #fill the tables
            self.quick_shl.append(fid.shl)
            self.fr_list.append(fid.parameters['FR'])
            try:
                self.temp_list.append(fid.parameters.get('_ITC_R1',0))
                self.temp_list2.append(fid.parameters.get('_ITC_R2',0))
            except:
                pass

        #set the frequency range
        self.fr_min, self.fr_max = (min(self.fr_list), max(self.fr_list))
        self.fr_step = (self.fr_max - self.fr_min)/len(self.file_list)

        #resort the filelist by frequency
        ###try fix frequency sorting (has to happen sooner!)
        sorting = np.argsort(self.fr_list)
        self.fr_list = np.array(self.fr_list)[sorting]
        self.file_list = np.array(self.file_list)[sorting]
        self.quick_shl = np.array(self.quick_shl)[sorting]
        try:
            self.temp_list = np.array(self.temp_list)[sorting]
            self.temp_list2 = np.array(self.temp_list2)[sorting]
        except: pass

        return (self.temp_list, self.temp_list2, self.temp_set, self.fr_list, self.quick_shl)

            

class T1:
    '''Evaluates a series of fid measurements, fitting T1'''

    def __init__(self, file_key, file_dir):
        '''Initializes the class and sets the file keys and directory'''
        self.file_key = file_key
        self.file_dir = file_dir
        self.Find_files()

        self.analysed = False
        self.disabled = False
        self.mirroring = False

        self.Get_params()

    def Reinit(self):
        '''deletes all content and reinitializes class'''
        file_key = self.file_key
        file_dir = self.file_dir
        #clear parameters and restart
        self.__dict__.clear()
        self.__init__(file_key, file_dir)

    def Find_files(self):
        '''Makes a list of all files with key and 000-999.DAT'''
        dir_list = os.listdir(self.file_dir)
        key = '^' + self.file_key + '-[0-9]*.DAT$'
        #save the sorted list
        def Sort_key(item):
            '''sort key to sort files by last number xxx.DAT'''
            return int(item.split('-')[-1][:-4])
        self.file_list = sorted([i for i in dir_list if re.search(key, i)], key=Sort_key)


    def Run(self, shl_convol=1):
        '''Uses the determined ranges and finalizes calculations'''
        self.area_list = list()

        #perhaps mean phase should be recalculated!!
        #allow for running with fid integral?

        for file in self.file_list:
            fid = FID(file, self.file_dir)
            fid.Offset(self.offset_range)
            fid.Shift_left(self.mean_shl, mirroring=self.mirroring)
            fid.Fourier()
            fid.Phase_rotate(self.mean_phase)
            fid.Integral_spc(self.integral_range)
            #update lists
            self.area_list.append(fid.area_spc)

    def Extract_T1(self):
        '''Runs all the routines required to get the T1 of the series'''
        #prepare data lists later save as np.array
        tau_list = list()
        temp_list = list()
        area_list = list()
        shl_list = list()
        phase_list = list()
        #get mean analysis values from last few points in set
        self.Get_means()
        #get points
        for file in self.file_list:
            fid=FID(file, self.file_dir)
            fid.Offset(offset_range=(-5,None))
            fid.Find_SHL()
            fid.Shift_left(self.shl_mean)
            fid.Fourier()
            fid.Phase_spc()
            fid.Phase_rotate()
            fid.Integral_spc()
            #update lists
            area_list.append(fid.area)
            tau_list.append(fid.parameters['D5'])
            shl_list.append(fid.shl)
            phase_list.append(fid.phase)
            if self.temp_set > 49:
                self.temp_list.append(fid.parameters.get('_ITC_R1',0))
            else:
                self.temp_list.append(fid.parameters.get('_ITC_R2',0))

    def Get_params(self):
        '''Extracts useful constant parameters from FIDs and saves into trace'''

        #initiate a representable FID
        fid = FID(self.file_list[-1], self.file_dir)
        #copy other usefull values
        self.TAU = str(1000000*fid.parameters['TAU'])+'u'
        self.D1 = str(1000000*fid.parameters['D1'])+'u'
        self.D2 = str(1000000*fid.parameters['D2'])+'u'
        self.D3 = str(1000000*fid.parameters['D3'])+'u'
        self.D9 = str(int(1000*fid.parameters['D9']))+'m'
        self.NS = int(fid.parameters['NS'])

        try:
            self.D5_min = str(1000000*self.tau_list[0])+'u'
        except: print("tau_list doesnt exist yet")
     
     
    def Get_means(self, mean_range=(-5,None), offset_range=(-200,None), phase_range=(0,-1)):
        '''Gets the mean phases and SHL from last 4 points'''
        #initialize mean counters
        phase_mean = 0
        shl_mean = 0
        n = 0
        #go over the selected files
        for file in self.file_list[slice(*mean_range)]:
            fid=FID(file, self.file_dir)
            fid.Offset(offset_range)
            fid.Find_SHL()
            fid.Shift_left(fid.shl)
            fid.Fourier()
            fid.Phase_spc(phase_range)
            #update mean values
            phase_mean += fid.phase_spc
            shl_mean += fid.shl
            n += 1
        #save means
        self.phase_mean = phase_mean / n
        self.shl_mean = shl_mean / n

    def Quick_T1(self, offset_range=(-200,-1), phase_range=(0,-1), shl_convol=1, integral_range=None):
        '''Runs through all files to get the T1 trend and phase and SHL values'''
     
        self.quick_T1 = list()
        self.quick_phase = list()
        self.quick_shl = list()
        self.temp_list = list()
        self.temp_list2 = list()
        self.tau_list = list()

        fid = FID(self.file_list[-1], self.file_dir)
        self.temp_set = fid.parameters.get('_ITC_R0',0)
        self.fr = fid.parameters['FR']

        #integral range hevristics
        if not integral_range:
            integral_range = (int(fid.parameters['D3']/fid.parameters['DW']/2),
                              int(fid.parameters['D3']/fid.parameters['DW']*2))

        for file in self.file_list:
            fid = FID(file, self.file_dir)
            fid.Offset(offset_range)
            fid.Find_SHL(convolution=shl_convol)
            fid.Shift_left(fid.shl)
            fid.Fourier()
            fid.Phase_spc(phase_range)
            fid.Phase_rotate(fid.phase_spc)
            fid.Integral_fid(integral_range)
            #fill the tables
            self.tau_list.append(fid.parameters['D5'])
            self.quick_T1.append(fid.area_fid)
            self.quick_phase.append(fid.phase_spc)
            self.quick_shl.append(fid.shl)
            self.temp_list.append(fid.parameters.get('_ITC_R1',0))
            self.temp_list2.append(fid.parameters.get('_ITC_R2',0))

        #resort the filelist by D5
        sorting = np.argsort(self.tau_list)
        self.tau_list = np.array(self.tau_list)[sorting]
        self.file_list = np.array(self.file_list)[sorting]
        self.quick_T1 = np.array(self.quick_T1)[sorting]
        self.quick_shl = np.array(self.quick_shl)[sorting]
        self.quick_phase = np.array(self.quick_phase)[sorting]
        try:
            self.temp_list = np.array(self.temp_list)[sorting]
            self.temp_list2 = np.array(self.temp_list2)[sorting]
        except: pass


        return (self.temp_list, self.temp_list2, self.temp_set, self.tau_list, self.quick_T1, self.quick_phase, self.quick_shl)
        

class T2:
    '''Evaluates a series of fid measurements, analyzing T2'''

    def __init__(self, file_key, file_dir):
        '''Initializes the class and sets the file keys and directory'''
        self.file_key = file_key
        self.file_dir = file_dir
        self.Find_files()

        self.analysed = False
        self.disabled = False
        self.mirroring = False

        self.Get_params()

    def Reinit(self):
        '''deletes all content and reinitializes class'''
        file_key = self.file_key
        file_dir = self.file_dir
        #clear parameters and restart
        self.__dict__.clear()
        self.__init__(file_key, file_dir)

    def Find_files(self):
        '''Makes a list of all files with key and 000-999.DAT'''
        dir_list = os.listdir(self.file_dir)
        key = '^' + self.file_key + '-[0-9]*.DAT$'
        #save the sorted list
        def Sort_key(item):
            '''sort key to sort files by last number xxx.DAT'''
            return int(item.split('-')[-1][:-4])
        self.file_list = sorted([i for i in dir_list if re.search(key, i)], key=Sort_key)


    def Get_params(self):
        '''Extracts useful constant parameters from FIDs and saves into trace'''
        if self.file_list == []:
            print(self.file_key)
        #initiate a representable FID
        fid = FID(self.file_list[-1], self.file_dir)
        #copy other usefull values
        self.D1 = str(1000000*fid.parameters['D1'])+'u'
        self.D3 = str(1000000*fid.parameters['D3'])+'u'
        self.D9 = str(int(1000*fid.parameters['D9']))+'m'
        self.NS = int(fid.parameters['NS'])


    def Quick_T2(self, offset_range=(-200,-1), phase_range=(0,-1), shl_convol=1, integral_range=None):
        '''Quick T2 points for range and mean selections'''
            
        self.quick_T2 = list()
        self.quick_phase = list()
        self.quick_shl = list()
        self.temp_list = list()
        self.temp_list2 = list()
        self.tau_list = list()

        fid = FID(self.file_list[0], self.file_dir)
        self.temp_set = fid.parameters.get('_ITC_R0',0)
        self.fr = fid.parameters['FR']

        #integral range hevristics
        if not integral_range:
            integral_range = (int(fid.parameters['D3']/fid.parameters['DW']/2),
                              int(fid.parameters['D3']/fid.parameters['DW']*2))

        for file in self.file_list:
            fid = FID(file, self.file_dir)
            fid.Offset(offset_range)
            fid.Find_SHL(convolution=shl_convol)
            fid.Shift_left(fid.shl)
            fid.Fourier()
            fid.Phase_spc(phase_range)
            fid.Phase_rotate(fid.phase_spc)
            fid.Integral_fid(integral_range)
            #fill the tables
            self.tau_list.append(fid.parameters['TAU'])
            self.quick_T2.append(fid.area_fid)
            self.quick_phase.append(fid.phase_spc)
            self.quick_shl.append(fid.shl)
            self.temp_list.append(fid.parameters.get('_ITC_R1',0))
            self.temp_list2.append(fid.parameters.get('_ITC_R2',0))


        #resort the filelist by tau
        sorting = np.argsort(self.tau_list)
        self.tau_list = np.array(self.tau_list)[sorting]
        self.file_list = np.array(self.file_list)[sorting]
        self.quick_T2 = np.array(self.quick_T2)[sorting]
        self.quick_shl = np.array(self.quick_shl)[sorting]
        self.quick_phase = np.array(self.quick_phase)[sorting]
        try:
            self.temp_list = np.array(self.temp_list)[sorting]
            self.temp_list2 = np.array(self.temp_list2)[sorting]
        except: pass

        return (self.temp_list, self.temp_list2, self.temp_set, self.tau_list,
                self.quick_T2, self.quick_phase, self.quick_shl)
        
    def Run(self):
        '''Uses the determined ranges and finalizes calculations'''
        self.area_list = list()

        #perhaps mean phase should be recalculated!!
        #allow for running with fid integral?

        for file in self.file_list:
            fid = FID(file, self.file_dir)
            fid.Offset(self.offset_range)
            fid.Shift_left(self.mean_shl, mirroring=self.mirroring)
            fid.Fourier()
            fid.Phase_rotate(self.mean_phase)
            fid.Integral_spc(self.integral_range)
            #update lists
            self.area_list.append(fid.area_spc)


class D1:
    '''Evaluates a series of fid measurements, analyzing D1 maximum'''

    def __init__(self, file_key, file_dir):
        '''Initializes the class and sets the file keys and directory'''
        self.file_key = file_key
        self.file_dir = file_dir
        self.Find_files()

        self.analysed = False
        self.disabled = False


    def Find_files(self):
        '''Makes a list of all files with key and 000-999.DAT'''
        dir_list = os.listdir(self.file_dir)
        key = '^' + self.file_key + '-[0-9]*.DAT$'
        #save the sorted list
        def Sort_key(item):
            '''sort key to sort files by last number xxx.DAT'''
            return int(item.split('-')[-1][:-4])
        self.file_list = sorted([i for i in dir_list if re.search(key, i)], key=Sort_key)


    def Extract_D1(self):
        '''Runs all the routines required to get the T2 of the series'''
        pass

    

class Series():
    '''Series, replaces separate series classes'''
    def __init__(self, parent, series_type):
        '''initializes the series class and remembers the type relevant info'''
        # 'calling_name': [class_for_point, file_prepend] #
        types = {'T1vT': [T1, 'T1'], 'T2vT': [T2, 'T2'], 'Spectrum': [Glue_spc, 'spc'], 'D1_sweep': [D1, 'D1']}
        self.file_prepend = types[series_type][1]
        self.point_method = types[series_type][0]

        #reference to parent (experiment_data class)
        self.parent = parent
        #dictionary of traces, data: (make sure it never gets deleted!)
        self.traces = dict()
        self.Keys()


    def Keys(self):
        '''finds the unique keys from parent and adds to the trace structure'''
        self.points = [point for point in self.parent.raw_file_list if point[0][0]==self.file_prepend]
        self.keys = sorted(list(set(point[1] for point in self.points)))
        #make sure there are trace entries for each key
        for key in self.keys:
            if key not in self.traces:
                self.traces[key] = dict()
        #adds all the new points in the tree
        for point in self.points:
            key = point[1]
            temp = point[0][-1]
            if temp not in self.traces[key]:
                #creates runs the T1 class for the selected temperature
                self.traces[key][temp] = self.point_method(point[2], point[3])
        
        
class Test():
    '''Dummy testing class'''
    def __init__(self,parent):
        self.parent = parent


class Experiment_data():
    '''A class that handles data searching, adding and saving'''
    def __init__(self, experiment):
        '''Prepares the selected experiment'''
        #define directories
        self.file_dir = os.path.join('data', experiment)
        #self.raw_dir = os.path.join('data', experiment, 'raw')
        self.pkl_dir = os.path.join('data', experiment, 'pkl')

        #self.possible_series = {'Spectrum':Spectrum, 'T1vT':T1vT, 'T2vT':T2vT}
        self.possible_series = ['Spectrum', 'T1vT', 'T2vT', 'D1_sweep']

        #Flag if data is loaded
        self.opened = False


    def Add_series(self):
        '''Finds all files and makes the predefined series classes. This will DELETE all previous data!!!'''
        #search for files
        self.Find_raw_files()
        #introduce the classes for further analysis        
        self.series = dict()
        #initialize all possible series
        for serie in self.possible_series:
            #self.series[serie]=self.possible_series[serie](self)
            self.series[serie]=Series(self,serie)

    def Pkl_load(self):
        '''Loads all pickled data'''
        #search the pkl dir for files, removes ending!
        pkl_list = [os.path.splitext(pkl)[0] for pkl in os.listdir(self.pkl_dir)if pkl.endswith('.pkl')]
        #adds all the pickled data into the class
        for p in pkl_list:
            p_name = os.path.join(self.pkl_dir, p + '.pkl')
            with open(p_name, 'rb') as pfile:
                self.__dict__[p] = pickle.load(pfile)
        #cringy fix for referencing (remind them who their parent is :)
        for serie in self.series:
            self.series[serie].parent = self

    def Pkl_save(self):
        '''Saves the data that should be pickled'''
        #save
        for p in GLOBAL_pkl_list:
            p_name = os.path.join(self.pkl_dir, p + '.pkl')
            with open(p_name, 'wb') as pfile:
                pickle.dump(self.__dict__[p], pfile, protocol=-1)

    #for now going with a linear list of filenames...
    #implement dictionaries if slow!!!
    def Find_raw_files(self):
        '''Makes a list of all raw files in directory and subdir'''
        raw_file_list = list()

        def Sort_key(item):
            '''Sorts over file keys first, then temperature'''
            l1 = len(item[0])
            #looks at the key split and takes other parts first and temperature last
            return (item[0][:-1],float(item[0][-1]))


        #takes the lists of directory path, directory name and the file name
        for dir_path, dir_names, file_names in os.walk(self.raw_dir):
            #takes every file matching the ending in directory
            for file_name in [file for file in file_names if file.endswith('G.DAT') or file.endswith('G.dat')]:
                split = file_name.split('-')[:-1]  #ignores xxx.DAT
                file_key = '-'.join(split)
                #put T at end
                for s in split:
                    if s[-1] == 'K':
                        split.remove(s)
                        #adds T as float
                        split.append(float(s[:-1].replace('p','.')))
                        break #ends once T is found
                unique_key = '-'.join(split[:-1])
                raw_file_list.append([split, unique_key, file_key, dir_path])
        #save list
        self.raw_file_list = sorted(raw_file_list, key=Sort_key)

    def File_sets(self):
        '''Shows the sets with different temperature from raw_file_list'''
        unique = sorted(list(set(['-'.join(i[0][:-1]) for i in self.raw_file_list])))
        for i in unique:
            print(i)




if __name__ == "__main__":
    '''Runs if this is the excecuted file'''
    #some test filenames
    fd = 'D:\Data\180716_CuIrO_NQR'
    fn = 'spc-18dB-20K-2lr-103.DAT'
        











