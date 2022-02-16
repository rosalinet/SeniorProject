
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from tkinter.constants import BOTTOM

from TkinterDnD2 import DND_FILES, TkinterDnD
from tkinter import *

###### Neural Network Imports

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split

import sys
sys.path.append("/path/to/script/file/directory/")



class Application(TkinterDnD.Tk):
    def __init__(self):
        super().__init__()

        self.title("Senior Project 2 GUI: CSV Reader/Protein SStype Neural Network Generator")
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand="true")
        self.geometry("900x700")
        self.search_page = SearchPage(parent=self.main_frame)
        
        


    

class DataTable(ttk.Treeview):
    def __init__(self, parent):
        super().__init__(parent)
        scroll_Y = tk.Scrollbar(self, orient="vertical", command=self.yview)
        scroll_X = tk.Scrollbar(self, orient="horizontal", command=self.xview)
        self.configure(yscrollcommand=scroll_Y.set, xscrollcommand=scroll_X.set)
        scroll_Y.pack(side="right", fill="y")
        scroll_X.pack(side="bottom", fill="x")
        self.stored_dataframe = pd.DataFrame()
        
        
        # Treeview
        self.neural_net = NeuralNetwork(parent)
        self.neural_net.place(rely=0.50, relx=0.25, relwidth=0.75, relheight=0.50)

    def set_datatable(self, dataframe):
        self.stored_dataframe = dataframe
        self._draw_table(dataframe.head(100))



    def _draw_table(self, dataframe):
        self.delete(*self.get_children())
        columns = list(dataframe.columns)
        self.__setitem__("column", columns)
        self.__setitem__("show", "headings")

        for col in columns:
            self.heading(col, text=col)

        df_rows = dataframe.to_numpy().tolist()
        for row in df_rows:
            self.insert("", "end", values=row)
        return None
    

    def find_value(self, pairs):
        # pairs is a dictionary
        new_df = self.stored_dataframe
        for col, value in pairs.items():
            query_string = f"{col}.str.contains('{value}')"
            new_df = new_df.query(query_string, engine="python")
        self._draw_table(new_df)


    def reset_table(self):
        self._draw_table(self.stored_dataframe)


class SearchPage(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        
        self.file_names_listbox = tk.Listbox(parent, selectmode=tk.SINGLE, bg="#D7A7AA", fg="white")
        self.file_names_listbox.place(relheight=1, relwidth=0.25)
        self.file_names_listbox.drop_target_register(DND_FILES)
        self.file_names_listbox.dnd_bind("<<Drop>>", self.drop_inside_list_box)
        self.file_names_listbox.bind("<Double-1>", self._display_file)
        self.file_names_listbox.opening = Label(self.file_names_listbox, text="Please drop your CSV files over here", background="#D7A7AA", fg="white", font=("Arial", 10)
              )
        self.file_names_listbox.opening.place(relx=1.0,rely=0.5,anchor=E)
        

        self.search_entrybox = tk.Entry(parent)
        self.search_entrybox.place(relx=0.25, relwidth=0.75)
        self.search_entrybox.bind("<Return>", self.search_table)

        # Treeview
        self.data_table = DataTable(parent)
        self.data_table.place(rely=0.05, relx=0.25, relwidth=0.75, relheight=0.45)

        self.path_map = {}

        

        

    def drop_inside_list_box(self, event):
        file_paths = self._parse_drop_files(event.data)
        current_listbox_items = set(self.file_names_listbox.get(0, "end"))
        for file_path in file_paths:
            if file_path.endswith(".csv"):
                path_object = Path(file_path)
                file_name = path_object.name
                if file_name not in current_listbox_items:
                    self.file_names_listbox.insert("end", file_name)
                    self.path_map[file_name] = file_path

    def _display_file(self, event):
        file_name = self.file_names_listbox.get(self.file_names_listbox.curselection())
        path = self.path_map[file_name]
        df = pd.read_csv(path, error_bad_lines=False, engine ='python')
      #  if not df.empty:
        #    Button(root, text="Choose Columns to Use", bg="#E19B9F", fg="white", command=null).pack(side=BOTTOM)
        self.data_table.set_datatable(df)
        self.data_table.neural_net.choose_columns(df)

    
    def get_dataframe(self, dataframe):
        return self.data_table



    def _parse_drop_files(self, filename):
        size = len(filename)
        res = []  # list of file paths
        name = ""
        idx = 0
        while idx < size:
            if filename[idx] == "{":
                j = idx + 1
                while filename[j] != "}":
                    name += filename[j]
                    j += 1
                res.append(name)
                name = ""
                idx = j
            elif filename[idx] == " " and name != "":
                res.append(name)
                name = ""
            elif filename[idx] != " ":
                name += filename[idx]
            idx += 1
        if name != "":
            res.append(name)
        return res

    def search_table(self, event):
        # column value. [[column,value],column2=value2]....
        entry = self.search_entrybox.get()
        if entry == "":
            self.data_table.reset_table()
        else:
            entry_split = entry.split(",")
            column_value_pairs = {}
            for pair in entry_split:
                pair_split = pair.split("=")
                if len(pair_split) == 2:
                    col = pair_split[0]
                    lookup_value = pair_split[1]
                    column_value_pairs[col] = lookup_value
            self.data_table.find_value(pairs=column_value_pairs)


class NeuralNetwork(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        #yscrollbar = Scrollbar(parent)
        #yscrollbar.pack(side = RIGHT, fill = Y)
        #UGLY CHECKBOX vvvv
        self.neural_window = tk.Listbox(parent, selectmode=tk.SINGLE, bg="#ecb7bf", fg="white")
        #BETTER LISTBOX vvvv
        #self.neural_window = tk.Listbox(parent, selectmode=MULTIPLE, bg="black", fg="white", yscrollcommand = yscrollbar.set)
        self.neural_window.place(rely=0.50, relx=0.25, relwidth=0.75, relheight=0.25)
        self.neural_window_bottom = tk.Frame(parent, bg="#Fbd2d7")
        self.neural_window_bottom.place(rely=0.75, relx=0.25, relwidth=0.75, relheight=0.25)
        
        #self.column_choices = set(["SStype"])
        #Application.add_scroll(self)

        
    def genLambda(self, col):
        return lambda: self.column_choices.add(col) 

    
   


    #LISTBOX VER
    def choose_columns(self, dataframe):
        #sbar = Scrollbar(self.neural_window, orient=VERTICAL, command=lbox.view).pack(side=RIGHT, fill=Y)
        lbox = Listbox(self.neural_window, selectmode=MULTIPLE, height=40, width=109, listvariable=StringVar(value=list(dataframe.columns)))
        lbox.pack(side='left', fill='y')
        sbar = Scrollbar(self.neural_window, orient=VERTICAL, command=lbox.yview)
        sbar.pack(side=RIGHT, fill=Y)
        lbox.config(yscrollcommand=sbar.set)
        
        Button(self.neural_window_bottom, text = "Run Neural Network with These Chosen Columns", height=5, width=70, command=lambda: self.neural_network(dataframe, lbox.curselection()), bg= "white", fg= "#CD5E77").pack(expand= YES)



    def neural_network(self, dataframe, column_choices):
        
        prot_features = dataframe.copy()
        choices = set()
        for x in column_choices:
            choices.add(dataframe.columns[x])
        choices.add("SStype")
        prot_features.drop(columns=filter(lambda x: x not in choices, prot_features.columns), inplace=True)
        prot_features.dropna(inplace=True)

        for choice in choices:
            if pd.api.types.is_string_dtype(prot_features[choice]):
                prot_features[choice] = prot_features[choice].astype('category').cat.codes

       
        train_data_df, val_data_df = train_test_split(prot_features, test_size=0.2)
        test_data_df, val_data_df = train_test_split(val_data_df, test_size=0.5)
        
        def breakdown(df : pd.DataFrame):
            df = df.copy()
            labels = df.pop("SStype")
            labels = labels.astype('category').cat.codes 
            return np.array(df), tf.keras.utils.to_categorical(labels)
        x_train, y_train = breakdown(train_data_df)
        x_test, y_test = breakdown(test_data_df)
        x_val, y_val = breakdown(val_data_df)

        normalize = preprocessing.Normalization()
        normalize.adapt(x_train)

        prot_model = tf.keras.Sequential([
            normalize,
            layers.Reshape((len(choices)-1, 1)),
            layers.Conv1D(64, 2, activation='relu'),
            layers.Conv1D(64, 2, activation='relu'),
            layers.Flatten(),
            layers.Dense(51, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(51, activation='relu'),
            layers.Dense(17, activation='relu'),
            layers.Dense(len(y_train[0]), activation='softmax')
            ])

        prot_model.compile(loss = tf.losses.CategoricalCrossentropy(), optimizer=tf.optimizers.Adam(), metrics=['categorical_accuracy'])
        prot_model.fit(x_train, y_train, epochs=100, batch_size=500, validation_data=(x_val, y_val), callbacks = [tf.keras.callbacks.EarlyStopping(patience = 4)])
        prot_model.evaluate(x_test, y_test, verbose=2)





if __name__ == "__main__":
    root = Application()
    root.mainloop()