from tkinter import *
from tkinter import ttk
from main import *
from tkinter import messagebox


def run_model():
     neurons = str(neurons_entry.get())
     neurons = neurons.split(",")
     neurons = [int(i) for i in neurons]

     layersNum = int(layers_entry.get())


     if len(neurons) != layersNum:
        messagebox.showinfo("Error ", "Number of layers doesn't match neuron numbers")
        return
     if(acitivationfun.current()==1):
        activation = "tangent"
     else:
        activation = "sigmoid"
     accuracy = Model(int(layers_entry.get()),neurons,float(eta_entry.get())
             , int(epochs_entry.get()), int(check_var.get()),activation)
     create_label("The Model Accuracy is: " + str(accuracy), 11, 1)


def testSample():
    if (acitivationfun.current() == 1):
        activation = "tangent"
    else:
        activation = "sigmoid"
    ID = userTesting(int(layers_entry.get()), activation,int(check_var.get()), float(feature1_entry.get()), float(feature2_entry.get()),
                     float(feature3_entry.get()), feature4_entry.current(), float(feature5_entry.get()))
    create_label("Sample Class Label is: " + ID, 12, 1)


def create_label(text, row, col):
    label = Label(window, text=text, font=40)
    label.grid(column=col, row=row, sticky=W, pady=10)


def create_cmb(row, col, values):
    feature_cmb = ttk.Combobox(window, values=values, width=20)
    feature_cmb['state'] = 'readonly'
    feature_cmb.grid(column=col, row=row)
    return feature_cmb


# create window
window = Tk()
window.title('Penguins model')
window.geometry('1100x700')


create_label("Enter number of hidden layers: ", 0, 0)
layers_entry = Entry(window, bd=5)
layers_entry.grid(row=0, column=1)

create_label("Enter number of neurons in each hidden layer: ", 1, 0)
neurons_entry = Entry(window, bd=5)
neurons_entry.grid(row=1, column=1)

create_label("Enter Learning rate: ", 2, 0)
eta_entry = Entry(window, bd=5)
eta_entry.grid(row=2, column=1)


create_label("Enter number of epochs: ", 3, 0)
epochs_entry = Entry(window, bd=5)
epochs_entry.grid(row=3, column=1)


# Choose Activation function
acitvationFunctions=("1- Sigmoid","2- Hyperbolic Tangent")
create_label("Select activation function: ", 4, 0)
acitivationfun = create_cmb(4, 1, acitvationFunctions)


# Add bias checkbox
check_var = IntVar()
bias_check = Checkbutton(window, variable=check_var, text="Add bias",
                         onvalue=1, offvalue=0, height=5, width=20, font=30)
bias_check.grid(row=5, column=1)


# create a button
btn = Button(window, text="Run Model", command=lambda : run_model(), font=30)
btn.grid(row=6, column=1)

# Classify a single sample
create_label("Enter bill length Value: ", 7, 0)
feature1_entry = Entry(window, bd=5)
feature1_entry.grid(row=7, column=1)

create_label("Enter bill depth Value: ", 7, 2)
feature2_entry = Entry(window, bd=5)
feature2_entry.grid(row=7, column=3)

create_label("Enter flipper length Value: ", 8, 0)
feature3_entry = Entry(window, bd=5)
feature3_entry.grid(row=8, column=1)

genders = ("Male","Female")
create_label("Select Gender: ", 8, 2)
feature4_entry = create_cmb(8, 3, genders)

create_label("Enter body mass Value: ", 9, 0)
feature5_entry = Entry(window, bd=5)
feature5_entry.grid(row=9, column=1)

# create a button
btn = Button(window, text="Test Sample", command=lambda : testSample(), font=30)
btn.grid(row=10, column=1)

window.mainloop()

