from __future__ import division
from tkinter import Tk, Label, Button, W, Entry, Text, END, messagebox
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import wget
import math


class text_GUI:
    def __init__(self, master):
        self.master = master
        self.text = None

        master.minsize(720, 400)
        master.title("Text generation app")

        self.label_greet = Label(master,
                           text="Enter data and press start to begin!", font=("Arial Bold", 12))
        self.label_greet.grid(row=0)

        self.label_epochs = Label(master,
                                  text="Number of training epochs (30 and above)")
        self.label_epochs.grid(row=1, sticky=W, padx=48)

        self.label_seed = Label(master,
                                text="Enter your seed text")
        self.label_seed.grid(row=2, sticky=W, padx=48)

        self.label_words = Label(master,
                 text="Number of words to generate")
        self.label_words.grid(row=3, sticky=W, padx=48)

        self.label_uploasd = Label(master,
                 text="Upload txt file with your seed txt")
        self.label_uploasd.grid(row=4, sticky=W, padx=48)

        self.label_output = Label(master,
                 text="Generated output text")
        self.label_output.grid(row=5, sticky=W, padx=48)

        self.e1 = Entry(master, bd=3)
        self.e2 = Text(master, bd=3, width=40, height=2)
        self.e3 = Entry(master, bd=3)

        self.e1.grid(row=1, column=1, pady=5)
        self.e2.grid(row=2, column=1, pady=5)
        self.e3.grid(row=3, column=1, pady=5)

        self.button = Button(master, text='Upload file', command=self.UploadAction)
        self.button.grid(row=4, column=1, sticky=W, padx=50)

        self.outputtext = Text(master, width=40, height=10, bd=4)
        self.outputtext.grid(column=1, row=5, pady=10)

        self.quit = Button(master,
                  text='Quit', command=master.quit, width=15)
        self.quit.grid(row=8, column=0, sticky=W, padx=50)

        self.start = Button(master,
                  text='Start', command=self.run, width=15)
        self.start.grid(row=8, column=1, sticky=W)

    def run(self):
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        print("Downloading data for training...")
        url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt"
        wget.download(url, out="/tmp/sonnets.txt")

        data = open('/tmp/sonnets.txt').read()

        corpus = data.lower().split("\n")

        tokenizer.fit_on_texts(corpus)
        total_words = len(tokenizer.word_index) + 1

        # create input sequences using list of tokens
        input_sequences = []
        for line in corpus:
            token_list = tokenizer.texts_to_sequences([line])[0]
            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i + 1]
                input_sequences.append(n_gram_sequence)

        # pad sequences
        max_sequence_len = max([len(x) for x in input_sequences])
        input_sequences = np.array(
            tf.keras.preprocessing.sequence.pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        # create predictors and label
        predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

        label = tf.keras.utils.to_categorical(label, num_classes=total_words)

        print("\nStarting training...")
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Embedding(total_words, 100, input_length=max_sequence_len - 1))  # Your Embedding Layer
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150, return_sequences=True)))  # An LSTM Layer
        model.add(tf.keras.layers.Dropout(0.5))  # A dropout layer
        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(150)))  # Another LSTM Layer
        model.add(tf.keras.layers.Dense(math.floor(total_words / 2), activation="relu",
                                        kernel_regularizer=tf.keras.regularizers.l2(
                                            0.01)))  # A Dense Layer including regularizers
        model.add(tf.keras.layers.Dense(total_words, activation="softmax"))  # A Dense Layer
        # Pick an optimizer

        model.compile(loss="categorical_crossentropy", optimizer="adam",
                      metrics=["accuracy"])  # Pick a loss function and an optimizer

        history = model.fit(predictors, label, epochs=int(self.e1.get()), verbose=1)

        messagebox.showinfo("Information", "Training finished.")

        if self.e2.get("1.0", "end-1c"):
            self.seed_text = self.e2.get("1.0", "end-1c")
        else:
            self.seed_text = self.text

        print("seed text", self.seed_text)

        self.next_words = int(self.e3.get())

        for _ in range(self.next_words):
            token_list = tokenizer.texts_to_sequences([self.seed_text])[0]
            token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len - 1,
                                                                       padding='pre')
            predicted = model.predict_classes(token_list, verbose=0)
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            self.seed_text += " " + output_word

        self.outputtext.insert(END, self.seed_text)

    def UploadAction(self):
        filename = filedialog.askopenfilename(filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
        print('Selected:', filename)
        f = open(filename, "r")
        self.text = "".join(line.rstrip() for line in f)


root = Tk()
my_gui = text_GUI(root)
root.mainloop()