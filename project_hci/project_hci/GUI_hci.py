import tkinter as tk
from tkinter import messagebox,ttk,filedialog
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle
#from help_fun import *
from fiducial_features import *
import warnings
warnings.filterwarnings('ignore')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)


def exe_GUI():

    top=tk.Tk()
    top.geometry('500x700')
    top.minsize(500,700) 
    top.maxsize(500,700)
    top.title('Authentication system GUI')
    top['background']='#856ff8'
    #opening file 
    #function definition for opening file dialog

    def openf():
        global file
        global to_test

        
        file = filedialog.askopenfilename(initialdir=r' D:\\Downloads\\project_hci', title="select file",
                                         filetypes=(("Txt Files",".txt"), ("All Files","*.*")))
        #inserting data to text editor
        to_test=np.loadtxt(file)
        msg=messagebox.showinfo('welcome','File inserted successfully')
        f_name=re.findall('sub_\w',file)

        teditor.delete('1.0', tk.END)
        teditor.insert(tk.END,f_name)
        print(to_test)
        

    #creating text editor
    teditor = tk.Text(top, width=16, height=2)
    teditor.place(x=200,y=50)
    # teditor.pack(pady=10)
    file_open = tk.Button(top, text="Open file", command= openf,width=12,height=2)
    file_open.pack()
    file_open.place(x=90,y=50)

    ###f2
    f2_l=tk.Label(top,text="Preprocessing",width=12,height=2)
    f2_l.place(x=90,y=100)


    f2_var=tk.StringVar()
    f2=ttk.Combobox(top,width=18,height=5,textvariable=f2_var)
    f2['values']=('pan_tompkins_11_point')
    f2.current(0)
    f2.place(x=200,y=100)

    def prep_type(type_):
        
      
        
        if type_ == 1:
            return  'pan_tompkins_11_point'
            
    def get_sub(idx):
        return f'sub_{idx+1}'



    def retrieve_f2():
        f2_v=f2.get()
        print(f2_v)
        return f2_v

    ##Run
    def plot(signal,test,type_):
        fig=Figure(figsize=(4,4),dpi=100)
        ax1=fig.add_subplot(2,1,1)
        ax1.set_title('signal',fontdict={'fontsize':12})
        ax1.plot(signal)

        ax2=fig.add_subplot(2,1,2)

 
        ax2.set_title(f'After preprocessing using {prep_type(type_)}',fontdict={'fontsize':12})
        ax2.plot(test)

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master = top)  
        canvas.get_tk_widget().place(x=50,y=300)
    
###########################
    lr=tk.Label(top,text="enter your Name[sub]",width=12,height=2)
    lr.place(x=90,y=5)
    lr_entry=tk.Entry(top,width=21)
    lr_entry.place(x=200,y=5)
    lr_entry.focus_set()
    lr_entry.insert(0,'sub_1')
    def callback_lr():
        lr_v=lr_entry.get()
        print(lr_v)
        return lr_v

################################

    def callback_Run():
        msg=messagebox.showinfo('Running!','Wait Please ')
        print("Code is Running")

        Output.delete('1.0', tk.END)
################################
        f2_v= retrieve_f2()
        type_=f2_v

     
        if type_=='pan_tompkins_11_point':
            type_=1


        # actual =re.findall('sub_\w',file)
        person_name=callback_lr()
        # print("actual",actual[0])


        test=preprocessing_11points(to_test)

        loaded_scalar=pickle.load(open("D:\\Downloads\\project_hci\\scalar.pkl",'rb'))
        test=loaded_scalar.transform(test)
        test=np.array(test[2])
        plot(to_test,test,type_)
        print('test shape',test.shape)
        print('test',test)



#######################

        # ('Wavelet','Fiducial','AC/DCT')
   
        # fiducial 
        if type_==1:
            loaded_model=pickle.load(open("D:\\Downloads\\project_hci\\LDA_classifier_11points.pkl",'rb'))
        
             
        if type_ ==1:
            test=np.expand_dims(test,axis=0)
        print('test shape after ',test.shape)

        pred=loaded_model.predict(test)

        print('prediction',pred[0])


        if person_name== get_sub(pred[0]):
            res='Subject identified --> Access Allowed'
        else:
            res='Subject is unidentified --> Access Denied'


        Output.insert(tk.END,res)
        print("Code is Running")

        

    B=tk.Button(top,text='Predict',height=2,width=10,command=callback_Run)
    B.place(x=225,y=150)

    Output = tk.Text(top, height = 2,
                  width = 35,
                  bg = "light cyan")
    Output.place(x=120,y=200)


    top.mainloop()
    
    
    
# In[42]:


def main():
    exe_GUI()

if __name__=='__main__':
    main()
