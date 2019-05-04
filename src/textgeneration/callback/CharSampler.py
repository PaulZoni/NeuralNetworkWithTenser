import os
import numpy as np
from keras.callbacks import Callback


class CharSampler(Callback):

    OUTPUT_FILE = '/Users/pavel/PycharmProjects/NeuralNetworkWithTenser/src/textgeneration/text/gen_text.txt'

    def __init__(self, char_vectors, model, num_chars, START_CHAR, indices_to_chars, END_CHAR):
        self.char_vectors = char_vectors
        self.model = model
        self.num_chars = num_chars
        self.START_CHAR = START_CHAR
        self.indices_to_chars = indices_to_chars
        self.END_CHAR = END_CHAR

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.epoch = 0
        if os.path.isfile(CharSampler.OUTPUT_FILE):
            os.remove(CharSampler.OUTPUT_FILE)

    def sample(self, preds, temperature=1.0):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def sample_one(self, T):
        result = self.START_CHAR
        while len(result) < 500:
            Xsampled = np.zeros((1, len(result), self.num_chars))
            for t, c in enumerate(list(result)):
                Xsampled[0, t, :] = self.char_vectors[c]
            ysampled = self.model.predict(Xsampled, batch_size=1)[0, :]
            yv = ysampled[len(result) - 1, :]
            selected_char = self.indices_to_chars[self.sample(yv, T)]
            if selected_char == self.END_CHAR:
                break
            result = result + selected_char
        return result

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = self.epoch + 1
        if self.epoch % 50 == 0:
            print('\nEpoch %d text sampling:' % self.epoch)
            with open(CharSampler.OUTPUT_FILE, 'a') as outf:
                outf.write('\n===== Epoch %d ====\n' % self.epoch)
                for T in [0.3, 0.5, 0.7, 0.9, 1.1]:
                    print('\tsampling, T = %.1f...' % T)
                    for _ in range(5):
                        self.model.reset_states()
                        res = self.sample_one(T)
                        outf.write('\nT = %.1f\n%s\n' % (T, res[1:]))
