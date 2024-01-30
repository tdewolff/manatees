import os
import sys
import time
import pickle
import string
import random
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torchaudio
from sklearn import metrics
import matplotlib.pyplot as plt

os.environ['TORCH_HOME'] = 'ast/pretrained_models'
sys.path.insert(1, './ast/src')
from models import ASTModel

parser = argparse.ArgumentParser("manatee training")
parser.add_argument("--epochs", help="Number of epochs to run training", type=int, default=0)
parser.add_argument("--split", help="Training/validation split", type=float, default=0.8)
parser.add_argument("--batch", help="Batch size", type=int, default=16)
parser.add_argument("--output", help="Output filename for trained model", type=str, default='model.pth')
parser.add_argument("data", nargs='*', help="Input filename for preprocessed data", type=str, default=['data.pkl'])
args = parser.parse_args()


# General settings
DataFilename = args.data
ModelFilename = args.output

# Data settings
FreqDims = 128
TimeDims = 32
WindowOverlap = 0.5
PositiveSplit = 0.5

# Training settings
TrainSplit = args.split
BatchSize = args.batch
LearningRate = 5e-5
#LearningRateEpochStart = 10
#LearningRateEpochStop = 1000
#LearningRateEpochStep = 5
#LearningRateDecay = 0.5
Epochs = args.epochs


def plot_spectrogram(spectrum):
    plt.matshow(spectral, aspect='auto', interpolation='nearest', origin='lower')
    plt.show()

uniq_id = ''.join(random.choice(string.ascii_lowercase + string.ascii_uppercase) for _ in range(6))
def uniq_model_filename():
    if ModelFilename == '':
        return ''
    idx = ModelFilename.rfind('.')
    if idx == -1:
        idx = len(ModelFilename)
    uniq = '-%s-%s' % (uniq_id, datetime.now().strftime('%Y%m%dT%H%M%S'))
    return ModelFilename[:idx] + uniq + ModelFilename[idx:]

def dur2str(dur):
    out = ''
    units = ['w', 'd', 'h', 'm', 's']
    scales = [7*24*3600, 24*3600, 3600, 60, 1]
    for i, unit in enumerate(units):
        n = dur // scales[i]
        if 0 < n:
            out += '%d%s' % (n, unit)
            dur = dur - n*scales[i]
    return out

def yield_samples(filename_data, metadata):
    print()
    print('Loading %s' % (filename_data,))
    signal, sample_rate = torchaudio.load(filename_data)
    if signal.shape[0] != 1:
        print('ERROR: audio file expected with 1 channel, got %d instead' % (filename_data, signal.shape[0]))
        return
    signal = signal - signal.mean()

    # get positive samples
    samples = []
    for i in range(len(metadata)):
        a = int(metadata[i,0]*float(sample_rate))
        b = int(metadata[i,1]*float(sample_rate)) + 1
        samples.append((i+1, 1, a, b))
    samples.sort(key=lambda x: x[2])

    i = 1
    while i < len(samples):
        if samples[i][2] < samples[i-1][3]:  # a(cur) < b(prev)
            print("WARNING: omitting sample %d which overlaps with sample %d" % (samples[i][0], samples[i-1][0]))
            samples.pop(i)
        else:
            i = i + 1

    num_positive = len(metadata)
    num_negative = int(float(num_positive)/PositiveSplit + 0.5) - num_positive

    # get negative samples randomly
    lengths = [b-a for (_, _, a, b) in samples]
    length_mean = np.mean(lengths)
    length_std = np.std(lengths)
    print("samples=%d length=%.3fs±%.3fs" % (num_positive, float(length_mean)/sample_rate, float(length_std)/sample_rate))
    for i in range(num_negative):
        # find negative sample randomly in audio file that doesn't overlap with other samples
        sample_pos = np.random.random()  # [0,1)
        sample_length = int(np.random.normal(length_mean, length_std) + 0.5)
        if sample_length <= 0:
            print("WARNING: random sample_length is negative for negative sample, are samples very short?")
            continue

        # find number of available start indices
        indices_used = sum([sample_length+(b-a) for (_, _, a, b) in samples])
        indices_available = signal.shape[1]-indices_used
        if indices_available <= 0:
            print("NOTICE: could not find space to insert negative sample")
            continue
        sample_pos = int(sample_pos*indices_available + 0.5)
        orig_pos = sample_pos

        # iterate over samples over audio file until we find our position
        start = 0
        inserted = False
        for i, (_, _, a, b) in enumerate(samples):
            end = a-sample_length
            if end < start:
                # too little space for sample
                continue
            elif start+sample_pos < end:
                # found position
                samples.insert(i, (-1, 0, start+sample_pos, start+sample_pos+sample_length))
                inserted = True
                break

            sample_pos = sample_pos - (end-start)
            start = b
        if not inserted:
            end = signal.shape[1]
            if start <= end and start+sample_pos < end:
                samples.insert(i, (0, 0, start+sample_pos, start+sample_pos+sample_length))
            else:
                print("NOTICE: could not find space to insert negative sample")

    for (_, cls, a, b) in samples:
        print("sample cls=%d time=%7.3fs—%7.3fs" % (cls, float(a)/sample_rate, float(b)/sample_rate))

        # calculate window size and shift in milliseconds to have TimeDims steps
        duration = float(b-a)*1000.0/sample_rate
        if WindowOverlap < 0.0 or 1.0 <= WindowOverlap:
            raise ValueError("WindowOverlap must be in [0,1)")
        shift = duration / (float(TimeDims) - float(WindowOverlap))
        length = (1.0-float(WindowOverlap))*shift

        # calculate fbank
        fbank = torchaudio.compliance.kaldi.fbank(
                signal[:,a:b], sample_frequency=sample_rate, num_mel_bins=FreqDims,
                window_type='hamming', frame_length=length, frame_shift=shift,
                htk_compat=True, channel=-1)
        if fbank.shape[0] != TimeDims or fbank.shape[1] != FreqDims:
            print("dur",duration, "shift",shift, "length",length, "sample_rate", sample_rate)
            raise ArithmeticError("unexpected fbank shape %s, expected %s" % (fbank.shape, [TimeDims, FreqDims]))

        yield {
            'input': fbank.numpy(),
            'target': torch.nn.functional.one_hot(torch.tensor(cls), 2).numpy(),
        }

if len(DataFilename) != 1 or os.path.isfile(DataFilename[0]):
    print('--- Loading preprocessed data')
    data = [sample for filename in DataFilename for sample in pickle.load(open(filename, 'rb'))]
else:
    print('--- Preprocessing data')
    data = []
    for i in range(1, 21):
        filename_data = f'data/Session{i}/Session{i}.wav'
        filename_metadata = f'data/Session{i}/Session{i}.Table.1.selections.txt'

        metadata = pd.read_csv(filename_metadata, sep='\t').values
        metadata = metadata[:,3:5]
        data.extend(list(yield_samples(filename_data, metadata)))
        print()
    if DataFilename != '':
        print('--- Saving preprocessed data')
        pickle.dump(data, open(DataFilename[0], 'wb'))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, device=None):
        # normalize dataset
        fbanks = np.array([sample['input'] for sample in data])
        mean, stddev = fbanks.mean(), fbanks.std()
        for i in range(len(data)):
            data[i]['input'] = (data[i]['input'] - mean) / (2*stddev)
            data[i]['input'] = torch.tensor(data[i]['input'], device=device, dtype=torch.float)
            data[i]['target'] = torch.tensor(data[i]['target'], device=device, dtype=torch.long)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]['input'], self.data[idx]['target']


# set up data loaders
print('\n--- Loading data')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dataset = Dataset(data, device=device)

num_data = len(dataset)
num_train = round(num_data * TrainSplit)
num_val = num_data - num_train
if num_val == 0:
    raise ValueError('validation set is empty')
print('Total samples: all=%d  train=%d  validation=%d' % (len(dataset), num_train, num_val))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BatchSize, shuffle=False)


# set up model
print('\n--- Loading model')
cwd = os.getcwd()
os.chdir('./ast/src/models')  # to download pretrained models
model = ASTModel(label_dim=2,
    fstride=10, tstride=10,
    input_fdim=FreqDims, input_tdim=TimeDims,
    imagenet_pretrain=True, audioset_pretrain=True,
    model_size='base384', verbose=False)
os.chdir(cwd)

model = model.to(device)


# training
parameters = [p for p in model.parameters() if p.requires_grad]
print('Total parameters: %.2f million' % (sum(p.numel() for p in parameters)/1e6,))
optimizer = torch.optim.Adam(parameters, LearningRate, weight_decay=5e-7, betas=(0.95, 0.999))
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=LearningRateDecay,
#        list(range(LearningRateEpochStart, LearningRateEpochStop, LearningRateEpochStep)))
criterion = torch.nn.CrossEntropyLoss()

if 0 < Epochs:
    print('\n--- Training')
    losses = []
    start_time = time.time()
    last_save = start_time

    print('Start training: %s' % (datetime.now(),))
    for epoch in range(Epochs):
        # training step
        model.train()
        epoch_losses = []
        for i, (inputs, targets) in enumerate(train_dataloader):
            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(targets, axis=1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            # scheduler.step()

            epoch_losses.append(float(loss))
        loss = np.array(epoch_losses).mean()

        # validation step
        model.eval()
        epoch_losses = []
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_dataloader):
                outputs = model(inputs)
                loss = criterion(outputs, torch.argmax(targets, axis=1))
                epoch_losses.append(float(loss))
        val_loss = np.array(epoch_losses).mean()

        print('%4d/%d   t=%5s   loss=%12g   val_loss=%12g' % (epoch+1, Epochs, dur2str(time.time()-start_time), loss, val_loss))
        if 3600.0 < time.time()-last_save and ModelFilename != '':
            torch.save(model.state_dict(), uniq_model_filename())
            last_save = time.time()

    print('End training: %s' % (datetime.now(),))
    if ModelFilename != '':
        torch.save(model.state_dict(), uniq_model_filename())


# validation
print('\n--- Validation')
val_losses = []
val_outputs = []
val_targets = []
model.eval()
with torch.no_grad():
    for i, (inputs, targets) in enumerate(val_dataloader):
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, torch.argmax(targets, axis=1))

        val_losses.append(float(loss))
        val_outputs.append(outputs)
        val_targets.append(targets)

print('\n--- Results')
loss = np.mean(val_losses)
print('Loss:           %g' % (loss,))

outputs = torch.cat(val_outputs).detach().cpu().numpy()
targets = torch.cat(val_targets).detach().cpu().numpy()
acc = metrics.accuracy_score(np.argmax(targets,1), np.argmax(outputs,1))
print('Accuracy:       %g' % (acc,))

aps = []
aucs = []
target_names = ['Negative', 'Positive']
for cls in range(2):
    aps.append(metrics.average_precision_score(targets[:,cls], outputs[:,cls], average=None))
    aucs.append(metrics.roc_auc_score(targets[:,cls], outputs[:,cls], average=None))
print('Avg. precision: %g' % (np.mean(aps),))
print('AUC:            %g' % (np.mean(aucs),))

report = metrics.classification_report(np.argmax(targets,1), np.argmax(outputs,1), target_names=target_names)
print()
print(report)
