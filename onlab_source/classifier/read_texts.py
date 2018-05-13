from __future__ import print_function
import os
import numpy as np
import gensim
import codecs
import csv
import re
#from gensim.models import KeyedVectors

np.random.seed(1337)

# Importok, köztük egy text tokenizáló (szavakra bont, kidobálja a felesleges karaktereket)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import sys

# beágyazásokat fogunk használni cikkek tematikus osztályozására
BASE_DIR = '.'
VECTORS_FILE = BASE_DIR + '/word2vec/w2v_model_index_nostem_size_100' # ebben a file-ban vannak a vektorok
TEXT_DATA_DIR = BASE_DIR + '/index/' # itt lesznek a minták
MAX_NB_WORDS = 20000 # Ennyi különböző szót kezelünk majd - ??
EMBEDDING_DIM = 100 # Ekkora lesz a használt beágyazás - elvileg 100 dimenzió lesz itt is
FILENAME_PREFIX = "index_crawl"

#model = Word2Vec.load(VECTORS_FILE)
#model = Word2Vec.load_word2vec_format(VECTORS_FILE)
model = gensim.models.Word2Vec.load(VECTORS_FILE)

#print(model['próba'])
#print("kész!")

texts = []
labels_index = {}
labels = []

#végigmegyünk a TEXT_DATA_DIR könyvtáron, beolvassuk az index_crawl kezdetű file-okat
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    fpath = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isfile(fpath) and name.startswith(FILENAME_PREFIX): # ha megfelelő prefixszel kezdődik, beolvassuk
        #f = open(fpath, encoding="utf-8")
        #with codecs.open(fpath, "r", encoding='utf-8', errors='ignore') as f:
        #    texts.append(f.read())
        #    print(name + " kész!")
        csvReader = csv.reader(codecs.open(fpath, 'rU', 'utf-8'))
        firstLine = True
        for row in csvReader:
            #első sort eldobjuk
            if firstLine:
                firstLine = False
                continue
            link = row[0] #megkaptuk a cikkeket, első elem a link, többi mehet text-be
            #regex-szel 
            m = re.search('index.hu/(.+?)/', link)
            if m:
                label_id = len(labels_index) #egész számot rendelünk az azonosítókhoz
                label_name = m.group(1) # a regexből kiolvasott "csoport"
                if label_name not in labels_index: # ha még nem adtunk azonosítót ennek a labelnek
                    labels_index[label_name] = label_id # akkor most adunk
                labels.append(label_id) # listában eltároljuk az egyes cikkekhez, hogy milyen label tartozik hozzájuk
            else:
                pass
			    
            # összefűzzük a maradék 3 elemet egy szöveggé - TODO: jó ez így??
            # még el kell távolítani a newline és a "\xa0" karaktereket a szövegből - \n, \xa0
            text = row[1] + " " + row[2] + " " + row[3]
            text = text.replace('\n', ' ')
            text = text.replace('\xa0', ' ')
            
            texts.append(text)
        print(name + " kész!")

print("==============================================")
print("Labels_index: " + str(labels_index) + "\n")
print("Labelindex size: " + str(len(labels_index)))
print("==============================================")
#23-ast akar, de 24-est kap

#first_text = list(csv.reader(szoveg, skipinitialspace=True))
#print (first_text[0:20]) # ez így karaktereket ad vissza
#print(list(csv.reader(szoveg, skipinitialspace=True))[0:4])

#
# A SZÖVEGMINTÁK VEKTORIZÁLÁSA
#

# Tokenizálás - szavak helyett indexek lesznek, ahol ugyanaz a szó szerepel, ott ugyanaz a szám...
# nem véletlenszerű, hogy milyen szám szerepel az egyes szavak helyén(?)!
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS) ## Tokenizál, legfeljebb MAX_NB_WORDS szóra
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts) # Soronként egy-egy szöveg szavai következnek egymás után listában.

word_index = tokenizer.word_index # különböző tokenek száma
print('Különböző szavak száma az összes szövegben: ', len(word_index))
print('tokenz: \n')
#print(word_index)

# ----------------------------------
import json

# as requested in comment
#word_index = {'word_index': word_index}

with open('target.txt', 'w') as file:
    txt = ""
    for key in word_index:
        row = key + " " + word_index[key] + "\n"
        txt += row
        
    file.write(json.dumps(txt)) # use `json.loads` to do the reverse

print('kééééééééééész!!!!')
     
# ----------------------------------

# Vágjuk / kiegészítjük mindet 1000 szó hosszúra
data = pad_sequences(sequences, maxlen=1000)

labels = to_categorical(np.asarray(labels))             # csinál egy bináris mátrixot??
print('A data tenzor alakja:', data.shape)
print('A label tenzor alakja:', labels.shape)

# Csinálunk egy validációs halmazt.
# Teszthalmazunk most nem lesz, a tanítás során a validációs halmazra kapunk majd accuracy értéket,
# ezt éles alkalmazásban ne csináljuk!

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(0.2 * data.shape[0])


# Keverés után marad egy tanító és egy validációs halmaz
x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples] # tréning halmaz
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:] # validációs halmaz

#
# BEÁGYAZÁS ALKALMAZÁSA A SZÖVEGRE
#
nb_words = min(MAX_NB_WORDS, len(word_index)) # Nincs értelme nagyobbra lőni, mint az adott szótárméret a GloVe-ban
embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM)) # Beágyazási mátrix, az importáltból fogjuk a sajátjunkat szemezgetni és átindexelni
for word, i in word_index.items():
    if i > MAX_NB_WORDS:
        continue
    #embedding_vector = embeddings_index.get(word)
    try:
        embedding_vector = model[word] # exception - word not in vocabulary...
    except KeyError:
        continue    # biztos jó???
    if embedding_vector is not None: # különben csupa nulla lesz a beágyazóvektor
        embedding_matrix[i] = embedding_vector
        
print ('A szövegre alkalmazott beágyazási mátrix sorainak száma:', len(embedding_matrix))

#
# BEÁGYAZÓ RÉTEG ELKÉSZÍTÉSE
#

embedding_layer = Embedding(nb_words + 1,
                            EMBEDDING_DIM, # Ilyen hosszúak a beágyazó vektoraink
                            weights=[embedding_matrix],
                            input_length=1000, # Ez az első 1000 szó minden szövegrészből
                            trainable=False) # Súlyokat nem hagyjuk módosítani


# Tanítunk egy 1d konvolúciós hálót osztályozásra
sequence_input = Input(shape=(1000,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences) # 128 5-ös szűrő fut végig a szövegeken háromszor
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          nb_epoch=2, batch_size=128)