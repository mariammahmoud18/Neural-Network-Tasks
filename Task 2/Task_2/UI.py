from tkinter import *
from tkinter import ttk
from Model import *
from tkinter import messagebox



def run_model():
    feature1 = first_feature.current()
    feature2 = sec_feature.current()
    #print(feature1)
    if feature1 == feature2:
        messagebox.showinfo("Error ", "The entered 2 features are the same!")
        return
    classes = selected_classes.current()
    label1 = 0
    label2 = 0
    if classes == 0:
        label1 = 1
        label2 = 2
    elif classes == 1:
        label1 = 1
        label2 = 3
    elif classes == 2:
        label1 = 2
        label2 = 3

    #checking bias
    if(int(check_var.get())):
       b = random.random()
    else:
        b = 0
    accuracy = Training(feature1+1, feature2+1, label1, label2, float(eta_entry.get())
             , int(epochs_entry.get()), b, float(MSE_entry.get()) )
    create_label("The Model Accuracy is: " + str(accuracy), 10, 1)
    if feature1_entry.get() and feature2_entry.get():
       result = userTesting(float(feature1_entry.get()), float(feature2_entry.get()), b)
       create_label("Class ID: " + str(result), 11, 1)




def create_label(text, row, col):
    label = Label(window, text=text, font=40)
    label.grid(column=col, row=row, sticky=W, pady=10)


def create_cmb(row, col, values):
    feature_cmb = ttk.Combobox(window, values=values, width=50)
    feature_cmb['state'] = 'readonly'
    feature_cmb.grid(column=col, row=row)
    return feature_cmb






# create window
window = Tk()
window.title('Penguins model')
window.geometry('800x700')

features = ('1- bill length', '2- bill depth', '3- flipper length', '4- gender', '5- body mass')
classes_combination = ('1- Adelie & Gentoo', '2- Adelie & Chinstrap', '3- Gentoo & Chinstrap')

create_label("Select feature 1: ", 0, 0)
first_feature = create_cmb(0, 1, features)


create_label("Select feature 2: ", 1, 0)
sec_feature = create_cmb(1, 1, features)

create_label("Select 2 Classes: ", 2, 0)
selected_classes = create_cmb(2, 1, classes_combination)

create_label("Enter Learning rate: ", 3, 0)
eta_entry = Entry(window, bd=5)
eta_entry.grid(row=3, column=1)


create_label("Enter number of epochs: ", 4, 0)
epochs_entry = Entry(window, bd=5)
epochs_entry.grid(row=4, column=1)

create_label("Enter MSE threshold: ", 5, 0)
MSE_entry = Entry(window, bd=5)
MSE_entry.grid(row=5, column=1)

# Add bias checkbox
check_var = IntVar()
bias_check = Checkbutton(window, variable=check_var, text="Add bias",
                         onvalue=1, offvalue=0, height=5, width=20, font=30)
bias_check.grid(row=6, column=1)

#getting user values for single sample test
create_label("Enter 1st Feature Value: ", 7, 0)
feature1_entry = Entry(window, bd=5)
feature1_entry.grid(row=7, column=1)

create_label("Enter 2nd Feature Value: ", 8, 0)
feature2_entry = Entry(window, bd=5)
feature2_entry.grid(row=8, column=1)


# create run model button
btn = Button(window, text="Run Model", command=run_model, font=30)
btn.grid(row=9, column=1)


window.mainloop()

