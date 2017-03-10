from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle

from utils import TextLoader
from model import Model

from tkinter import *

# list to store preprocessed sentences from madlib styled template 
sentences = list()

# file to save generated sentences
fileToSave = "placeholder.txt"

# directory where to load trained models from
# save_shake save_WP1 save_WP2 save_Trump save_TOTC
save_dir_arg = "save_WP2"

# load saved arguments, word/vocab lists, and model to use
with open(os.path.join(save_dir_arg, 'config.pkl'), 'rb') as f:
	saved_args = cPickle.load(f)
with open(os.path.join(save_dir_arg, 'words_vocab.pkl'), 'rb') as f:
	words, vocab = cPickle.load(f)
model = Model(saved_args, True)

# TO-DO: user gets to load whichever model they want to use
def loadModel():
	content = entry_2.get()
	print(content)

# load a txt file into the GUI, txt file contains the template
def loadTemplate():
	# retrieve name of txt file containing madlib
	content = entry_1.get()
	directory = "data/" + content

	# prepare file to save
	global fileToSave
	fileToSave = directory.replace(".txt", "_filled.txt")

	# delete all text currently loaded into the text module in GUI
	text_1.delete(0.0, END)

	# open file in read-only mode
	f = open(directory, 'r')

	# get one line (sentence) at a time
	for idx, line in enumerate(f):
		# last index of space character
		lastIndex = line.rfind(' ') + 1

		# prepare sentences for GUI and neural net

		# get rid of underscores, prepare for formatting
		line = line.replace("_________.", "         .")

		# instead of underscores, we will underline text in the GUI
		# number before '.' represents row of sentence
		# number after '.' represents column of sentence 
		underlineIndexStart = str(idx+1) + "." + str(lastIndex)
		underlineIndexEnd = str(idx+1) + "." + str(len(line))

		# insert current sentence into text module 
		text_1.insert(idx+1.0, line)

		# add and configure tag for underlined text
		text_1.tag_add("underline", underlineIndexStart, underlineIndexEnd)
		text_1.tag_config("underline", background="gray", underline=1)

		# final processing steps before storing sentences for NN
		line = line.replace("         .", "")
		line = line.rstrip()
		line = line.replace('\n','')

		sentences.append(line)

	# close file
	f.close()

# generate 1 word answers
def sample():
	# default parameters
	#save_dir_arg = "save_WP2"  # default: "save"
	n_arg = 1  # default: 200
	#prime_arg = textInput  # default: " "
	pick_arg = 1  # default: 1
	sample_arg = 1  # default: 1

	# delete all text currently loaded into the text module in GUI
	text_1.delete(0.0, END)

	# open file to write generated sentences
	f = open(fileToSave, 'w')

	# start session
	with tf.Session() as sess:
		# initialize variables
		tf.global_variables_initializer().run()
		# create a saver
		saver = tf.train.Saver(tf.global_variables())
		# get checkpoint
		ckpt = tf.train.get_checkpoint_state(save_dir_arg)
		if ckpt and ckpt.model_checkpoint_path:
			# restore checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)
			
			# sample one sentence at a time
			for idx, line in enumerate(sentences):

				# print to terminal for debugging
				print("[" + line + "]")

				# sample sentence, receive generated output
				generatedSentence = model.sample(sess, words, vocab, n_arg, line, sample_arg, pick_arg)

				# clean up sentence for GUI
				generatedSentence += ".\n"
				generatedSentence = generatedSentence.replace("..",".")
				generatedSentence = generatedSentence.replace(",.",".")
				generatedSentence = generatedSentence.replace("?.",".")
				generatedSentence = generatedSentence.replace("!.",".")
				generatedSentence = generatedSentence.replace(";.",".")
				generatedSentence = generatedSentence.replace(":.",".")

				# where to underline and highlight generated words in sentence
				# number before '.' represents row of sentence
				# number after '.' represents column of sentence 
				lastIndex = generatedSentence.rfind(' ') + 1
				underlineIndexStart = str(idx+1) + "." + str(lastIndex)
				underlineIndexEnd = str(idx+1) + "." + str(len(generatedSentence)-2)

				# print to terminal for debugging
				print(idx+1.0, lastIndex, underlineIndexStart, underlineIndexEnd, generatedSentence)

				# insert sentence to text module in GUI
				text_1.insert(idx+1.0, generatedSentence)

				# underline generated text
				text_1.tag_add("underline", underlineIndexStart, underlineIndexEnd)
				text_1.tag_config("underline", background="yellow", underline=1)
				
				# write complete sentence to file
				f.write(generatedSentence)

	# close file
	f.close()

	# delete contents of sentences, from all locations, for next run
	del sentences[:]

# generate random text
def sample2():
	# default parameters
	#save_dir_arg = "save_WP2"  # default: "save"
	n_arg = 200  # default: 200
	prime_arg = " "  # default: " "
	pick_arg = 1  # default: 1
	sample_arg = 1  # default: 1

	# delete all text currently loaded into the text module in GUI
	text_1.delete(0.0, END)

	# start session
	with tf.Session() as sess:
		# initialize variables
		tf.global_variables_initializer().run()
		# create a saver
		saver = tf.train.Saver(tf.global_variables())
		# get checkpoint
		ckpt = tf.train.get_checkpoint_state(save_dir_arg)
		if ckpt and ckpt.model_checkpoint_path:
			# restore checkpoint
			saver.restore(sess, ckpt.model_checkpoint_path)

			# sample with random prime, receive generated output (200 words)
			generatedText = model.sample(sess, words, vocab, n_arg, prime_arg, sample_arg, pick_arg)

			# print to terminal for debugging
			print(generatedText)

			# insert text to text module in GUI
			text_1.insert(1.0, generatedText)

# set root for GUI using TKinter
root = Tk()

# name of GUI
root.title("MadLibs")

# size of GUI
root.geometry("1100x450")

# entry module to type template name to load
entry_1 = Entry(root)
entry_1.place(x = 20, y = 40)

# button to load template, calls function "loadTemplate"
button_1 = Button(root, text = "Load Template", command = loadTemplate, width = 10)
button_1.place(x = 215, y = 40)

# text field module to display generated output
text_1 = Text(root, width = 100, height = 20, wrap=WORD, bg="gray")
text_1.place(x = 350, y = 40)

# button to generate 1 word answers, calls function "sample"
button_2 = Button(root, text = "Generate Answers!", command = sample)
button_2.place(x = 350, y = 350)

# button to generate random text, calls function "sample2"
button_4 = Button(root, text = "Generate Text", command = sample2)
button_4.place(x = 500, y = 350)

# TO-DO: user gets to load whichever model they want to use
# entry module to type model name to load
#entry_2 = Entry(root)
#entry_2.place(x = 20, y = 80)

#button_3 = Button(root, text = "Load Model", command = loadModel, width = 10)
#button_3.place(x = 215, y = 80)

# start main loop for GUI
root.mainloop()
