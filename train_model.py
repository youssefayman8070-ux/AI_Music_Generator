import os
import pickle
import numpy as np

from music21 import converter, instrument, note, chord

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import to_categorical


DATASET_PATH = "midi_dataset"
MODEL_PATH = "models/music_model.keras"
NOTES_PATH = "models/notes.pkl"

SEQUENCE_LENGTH = 100


def get_notes():

    notes = []

    for root, dirs, files in os.walk(DATASET_PATH):

        for file in files:

            if file.endswith(".mid") or file.endswith(".midi"):

                path = os.path.join(root, file)

                print("Reading:", path)

                try:

                    midi = converter.parse(path)

                    parts = instrument.partitionByInstrument(midi)

                    if parts:
                        notes_to_parse = parts.parts[0].recurse()
                    else:
                        notes_to_parse = midi.flat.notes

                    for element in notes_to_parse:

                        if isinstance(element, note.Note):
                            notes.append(str(element.pitch))

                        elif isinstance(element, chord.Chord):
                            notes.append(".".join(str(n) for n in element.normalOrder))

                except Exception as e:
                    print("Error:", e)

    return notes


def prepare_sequences(notes):

    pitchnames = sorted(set(notes))

    note_to_int = dict(
        (note, number) for number, note in enumerate(pitchnames)
    )

    network_input = []
    network_output = []

    for i in range(0, len(notes) - SEQUENCE_LENGTH):

        sequence_in = notes[i:i + SEQUENCE_LENGTH]
        sequence_out = notes[i + SEQUENCE_LENGTH]

        network_input.append(
            [note_to_int[char] for char in sequence_in]
        )

        network_output.append(
            note_to_int[sequence_out]
        )

    n_patterns = len(network_input)

    network_input = np.reshape(
        network_input,
        (n_patterns, SEQUENCE_LENGTH, 1)
    )

    network_input = network_input / float(len(pitchnames))

    network_output = to_categorical(network_output)

    return network_input, network_output, pitchnames


def create_model(network_input, n_vocab):

    model = Sequential()

    model.add(
        LSTM(
            256,
            input_shape=(
                network_input.shape[1],
                network_input.shape[2]
            ),
            return_sequences=True
        )
    )

    model.add(Dropout(0.3))

    model.add(LSTM(256))

    model.add(Dense(256))

    model.add(Dropout(0.3))

    model.add(Dense(n_vocab))

    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    return model


if __name__ == "__main__":

    notes = get_notes()

    print("Total Notes:", len(notes))

    with open(NOTES_PATH, "wb") as filepath:
        pickle.dump(notes, filepath)

    network_input, network_output, pitchnames = prepare_sequences(notes)

    model = create_model(network_input, len(pitchnames))

    model.fit(
        network_input,
        network_output,
        epochs=20,
        batch_size=64
    )

    model.save(MODEL_PATH)

    print("Training Finished Successfully")