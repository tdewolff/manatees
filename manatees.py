import os
import sys
import time
import glob
import pickle
import signal
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

# exit cleanly from ctrl+c
signal.signal(signal.SIGINT, lambda x, y: sys.exit(1))


# TODO: find more datasets
# TODO: does it make sense to use the Mel scale since this is not human communication?


parser = argparse.ArgumentParser("Manatee model training and evaluation")
parser.add_argument("--epochs", help="Number of epochs to run training", type=int, default=3)
parser.add_argument("--split", help="Training/validation split", type=float, default=0.7)
parser.add_argument("--batch", help="Batch size", type=int, default=16)
parser.add_argument("--lr", help="Initial learning rate", type=float, default=1e-6)
parser.add_argument("--lr-step", help="Learning rate scheduler epoch step", type=int, default=5)
parser.add_argument("--lr-decay", help="Learning rate scheduler step decay", type=float, default=0.5)
parser.add_argument("--model", help="Output filename for trained model", type=str, default='model.pth')
parser.add_argument("--pos-split", help="Percentage of positive samples (by adding negative samples)", type=float, default=0.5)
parser.add_argument("--data", nargs='*', help="Input filename for preprocessed data", type=str, default=['data.pkl'])
parser.add_argument("--sound", nargs='*', help="Input filename for sound file for evaluation", type=str, default=[])
parser.add_argument("--report", help="Report filename", type=str, default='report.pkl')
parser.add_argument("cmd", nargs='?', choices=['train', 'test', 'eval'], type=str, default='eval')
args = parser.parse_args()


# General settings
Cmd = args.cmd
DataFilename = args.data
SoundFilename = args.sound
ModelFilename = args.model
ReportFilename = args.report

# Data settings
FreqDims = 64           # increases frequency resolution for each sample's fbank
TimeDims = 128          # increases time resolution for each sample's fbank
WindowOverlap = 0.5     # increases FFT window size but near time samples look more alike
SampleDuration = 1.0    # duration to use for each sample in seconds
SampleOverlap = 0.5     # overlap between samples for evaluation
PositiveSplit = args.pos_split     # part of the data set that is a positive sample
Balance = False

# Training settings
TrainSplit = args.split
BatchSize = args.batch
LearningRate = args.lr
LearningRateStep = args.lr_step
LearningRateDecay = args.lr_decay
Epochs = args.epochs

if abs(PositiveSplit) < 1e-6:
    print('ERROR: PositiveSplit is too low')
    exit(1)

# Utility functions
def uniq_filename(filename):
    idx = filename.rfind('.')
    if idx == -1:
        idx = len(filename)
    timestamp = '-' + datetime.now().strftime('%Y%m%dT%H%M%S')
    return filename[:idx] + timestamp + filename[idx:]

def dur2str(dur):
    out = ''
    units = ['w', 'd', 'h', 'm']
    scales = [7*24*3600, 24*3600, 3600, 60]
    for i, unit in enumerate(units):
        n = dur // scales[i]
        if 0 < n:
            out += '%d%s' % (n, unit)
            dur = dur - n*scales[i]
    out += '%ds' % (int(dur+0.5),)
    return out

def get_fbank(signal, sample_rate):
    # calculate window size and shift in milliseconds to have TimeDims steps
    duration = float(signal.shape[1])*1000.0/sample_rate
    if WindowOverlap < 0.0 or 1.0 <= WindowOverlap:
        raise ValueError("WindowOverlap must be in [0,1)")
    shift = duration / float(TimeDims)
    length = (1.0 + float(WindowOverlap))*shift

    # filter on frequencies emitted by manatees
    low_freq = 2000
    high_freq = 20000

    # calculate fbank
    fbank = torchaudio.compliance.kaldi.fbank(
            signal, sample_frequency=sample_rate, num_mel_bins=FreqDims,
            window_type='hamming', frame_length=length, frame_shift=shift,
            low_freq=low_freq, high_freq=high_freq,
            htk_compat=True, channel=-1)
    if fbank.shape[0] != TimeDims or fbank.shape[1] != FreqDims:
        if TimeDims < fbank.shape[0]:
            half = int(float(fbank.shape[0]-TimeDims)/2.0)
            fbank = fbank[half:half+TimeDims,:]
        else:
            raise ArithmeticError("unexpected fbank shape %s, expected %s" % (fbank.shape, [TimeDims, FreqDims]))
    return fbank.numpy()

def plot_fbank(signal, rate):
    fbank = get_fbank(signal, rate)
    plt.matshow(fbank, aspect='auto', interpolation='nearest', origin='lower')
    plt.show()

def yield_samples(filename_data, metadata=None, balance=False):
    print('Loading %s' % (filename_data,))
    signal, rate = torchaudio.load(filename_data)
    if signal.shape[0] != 1:
        print('ERROR: audio file expected with 1 channel, got %d instead' % (filename_data, signal.shape[0]))
        return
    signal = signal - signal.mean()

    # merge overlapping samples
    if metadata is not None:
        i = 1
        metadata = metadata[metadata[:,0].argsort()] # sort on start time
        while i < len(metadata):
            if metadata[i,0] < metadata[i-1,1]:  # start(current) < end(previous)
                print("INFO: merging overlapping samples %d and %d" % (i-1, i))
                metadata[i-1,0] = np.min(metadata[i-1:i+1,0])
                metadata[i-1,1] = np.max(metadata[i-1:i+1,1])
                metadata = np.delete(metadata, i, axis=0)
            else:
                i = i + 1

    # calculate sample shift in seconds for a certain duration and desired overlap
    duration = signal.shape[1]/rate
    shift = SampleDuration*(1.0-SampleOverlap)
    n = int((duration-shift)/shift + 0.5)
    shift = (duration-SampleDuration)/float(n-1)

    # extract positive and negative samples
    samples = np.empty((n,4))
    for i in range(n):
        start = i*shift
        end = start + SampleDuration

        # extract class when training or testing
        cls = None
        if metadata is not None:
            cls = 0
            for j in range(len(metadata)):
                sample_start = metadata[j,0]
                sample_end = metadata[j,1]
                if sample_start < end and start < sample_end and 0.5 <= (min(end,sample_end)-max(start,sample_start))/(sample_end-sample_start):
                    # atleast half the sample is inside the window
                    cls = 1

        # align with temporal resolution (index into signal)
        a = np.round(start*rate)
        b = a + int(SampleDuration*rate + 0.5)

        # use loudness to select louder negative sample more often
        lufs = None
        sample_signal = signal[:,int(a):int(b)]
        if balance:
            lufs = torchaudio.functional.loudness(sample_signal, rate)

        # add to samples
        samples[i,:] = [cls, a, b, lufs]

    if balance:
        positives = samples[samples[:,0] == 1,:]
        negatives = samples[samples[:,0] == 0,:]

        num_positive = len(positives)
        num_negative = int(float(num_positive)/PositiveSplit + 0.5) - num_positive

        # pick a subset of the negative samples
        if num_negative < len(negatives):
            # use loudness to select louder negative sample more often
            p = negatives[:,3]
            p = p - np.min(p)
            p /= np.sum(p)
            indices = np.random.choice(len(negatives), num_negative, replace=False, p=p)
            negatives = negatives[indices,:]
        elif len(negatives) < num_negative:
            print('WARNING: not enough negative samples to satisfy PositiveSplit=%g' % (PositiveSplit,))

        samples = np.concatenate((positives, negatives))

    # return samples
    for (cls, a, b, _) in samples:
        sample_signal = signal[:,int(a):int(b)]
        sample_rate = rate
        sample_fbank = get_fbank(sample_signal, sample_rate) 
        target = np.array(0) if cls is None else torch.nn.functional.one_hot(torch.tensor(int(cls)), 2).numpy()
        position = (a/sample_rate, b/sample_rate)

        yield {
            'input': sample_fbank,
            'target': target,
            'position': position,
        }

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
        return self.data[idx]

def load_model(filename, device):
    print('\n--- Loading model')
    cwd = os.getcwd()
    os.chdir('./ast/src/models')  # to download pretrained models
    model = ASTModel(label_dim=2,
        fstride=10, tstride=10,
        input_fdim=FreqDims, input_tdim=TimeDims,
        imagenet_pretrain=True, audioset_pretrain=True,
        model_size='base384', verbose=False)
    os.chdir(cwd)

    if Cmd != 'train':
        if not os.path.isfile(filename):
            # find last trained model with datetime/uniqid in filename
            idx = filename.rfind('.')
            if idx == -1:
                idx = len(filename)
            models = glob.glob(filename[:idx] + '-*' + filename[idx:])
            if len(models) == 0:
                print("ERROR: model filename '%s' doesn\'t exist, you must first train a model" % (filename,))
                exit(1)
            models.sort(reverse=True)
            filename = models[0]
        print('Using pretrained model from %s' % (filename,))
        model.load_state_dict(torch.load(filename))
    model = model.to(device)

    parameters = [p for p in model.parameters() if p.requires_grad]
    print('Total parameters: %.2f million' % (sum(p.numel() for p in parameters)/1e6,))
    return model

def report(model, criterion, dataloader, name=''):
    cum_outputs = []
    cum_targets = []
    cum_loss = torch.zeros((), device=device, dtype=float)
    with torch.no_grad():
        n = len(dataloader)
        start_time = time.time()
        last_print = start_time
        print('Start: %s' % (datetime.now(),))
        for i, sample in enumerate(dataloader):
            inputs = sample['input']
            targets = sample['target']
            cum_targets.append(targets)

            outputs = model(inputs)
            loss = criterion(outputs, torch.argmax(targets, axis=1))
            cum_loss += loss.detach()

            outputs = torch.sigmoid(outputs)
            cum_outputs.append(outputs)

            if 15.0 < time.time()-last_print:
                print('%6d/%d   t=%5s' % (i+1, len(dataloader), dur2str(time.time()-start_time)))
                last_print = time.time()
        print('End: %s' % (datetime.now(),))
    loss = float(cum_loss)/float(len(dataloader.dataset))
    print('Loss:           %g' % (loss,))

    outputs = torch.cat(cum_outputs).detach().cpu().numpy()
    targets = torch.cat(cum_targets).detach().cpu().numpy()
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

    if ReportFilename != '':
        data = {
            'outputs': outputs,
            'targets': targets,
        }
        filename = ReportFilename
        if name != '':
            idx = filename.rfind('.')
            if idx == -1:
                idx = len(filename)
            filename = filename[:idx] + '-' + name + filename[idx:]
        pickle.dump(data, open(filename, 'wb'))


#########################################################################
#########################################################################
#########################################################################


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = torch.nn.CrossEntropyLoss(reduction='sum')

if Cmd == 'train':
    # load data
    if len(DataFilename) != 1 or os.path.isfile(DataFilename[0]):
        print('--- Loading preprocessed data')
        data = []
        for filename in DataFilename:
            print('Loading %s' % (filename,))
            data.extend(pickle.load(open(filename, 'rb')))
    else:
        print('--- Preprocessing data')
        print('PositiveSplit:', PositiveSplit)
        data = []
        for i in range(1, 21):
            filename_data = f'data/Session{i}/Session{i}.wav'
            filename_metadata = f'data/Session{i}/Session{i}.Table.1.selections.txt'
            metadata = pd.read_csv(filename_metadata, sep='\t').values
            metadata = metadata[:,3:5]
            data.extend(list(yield_samples(filename_data, metadata, balance=Balance)))
        if DataFilename[0] != '':
            print('\n--- Saving preprocessed data')
            pickle.dump(data, open(DataFilename[0], 'wb'))

    # set up data loaders
    print('\n--- Loading data')
    dataset = Dataset(data, device=device)

    num_data = len(dataset)
    num_train = round(num_data * TrainSplit)
    num_val = num_data - num_train
    if num_val == 0:
        raise ValueError('validation set is empty')

    num_pos = sum([torch.argmax(sample['target']) == 1 for sample in dataset])
    num_neg = num_data - num_pos
    print('Total samples: all=%d  train/val=%d/%d  pos/neg=%d/%d' % (len(dataset), num_train, num_val, num_pos, num_neg))

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [num_train, num_val])

    # set up weighted sampler to deal with class imbalance
    num_train_pos = sum([torch.argmax(sample['target']) == 1 for sample in train_dataset])
    num_train_neg = len(train_dataset) - num_train_pos
    #weight_pos = 0.5 / float(num_train_pos)
    #weight_neg = 0.5 / float(num_train_neg)
    #sample_weights = torch.tensor([weight_pos if torch.argmax(sample['target']) == 1 else weight_neg for sample in train_dataset])
    #sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, num_samples=len(train_dataset), replacement=False)

    # use class weights to deal with class imbalance
    #weights = torch.tensor([float(num_train_pos) / float(num_train_neg), 1.0], device=device, dtype=torch.half)
    train_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BatchSize)

    # load model
    model = load_model(ModelFilename, device)

    # training
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, LearningRate, weight_decay=5e-7, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
            list(range(LearningRateStep, 1000, LearningRateStep)),
            gamma=LearningRateDecay)

    if 0 < Epochs:
        print('\n--- Training')
        start_time = time.time()
        last_save = start_time
        cum_loss = torch.zeros((), device=device, dtype=float)

        print('Start: %s' % (datetime.now(),))
        for epoch in range(Epochs+1):
            # training step
            model.train()
            cum_loss.zero_()
            for i, sample in enumerate(train_dataloader):
                inputs = sample['input']
                targets = sample['target']

                # augment sample with random noise
                min_snr = 2.0
                max_snr = 10.0
                snr = min_snr + (max_snr-min_snr)*np.random.random()
                inputs += torch.normal(0.0, 1.0/snr, inputs.shape, device=device)

                # calculate loss
                outputs = model(inputs)
                loss = train_criterion(outputs, torch.argmax(targets, axis=1))
                cum_loss += loss.detach()

                # update weights
                if 0 < epoch:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            train_loss = float(cum_loss)/float(len(train_dataloader.dataset))

            # validation step
            model.eval()
            cum_loss.zero_()
            with torch.no_grad():
                for i, sample in enumerate(val_dataloader):
                    inputs = sample['input']
                    targets = sample['target']

                    outputs = model(inputs)
                    loss = criterion(outputs, torch.argmax(targets, axis=1))
                    cum_loss += loss.detach()
            val_loss = float(cum_loss)/float(len(val_dataloader.dataset))

            print('%4d/%d   t=%5s   lr=%g   loss=%12g   val_loss=%12g' % (epoch, Epochs, dur2str(time.time()-start_time), scheduler.get_last_lr()[0], train_loss, val_loss))
            if 3600.0 < time.time()-last_save and ModelFilename != '':
                torch.save(model.state_dict(), uniq_filename(ModelFilename))
                last_save = time.time()
            if 0 < epoch:
                scheduler.step()

        print('End: %s' % (datetime.now(),))
        if ModelFilename != '':
            torch.save(model.state_dict(), uniq_filename(ModelFilename))

    model.eval()

    print('\n--- Training report')
    report(model, criterion, train_dataloader, 'train')

    # validation
    print('\n--- Validation report')
    report(model, criterion, val_dataloader, 'val')

    #print('\n--- Testing report')
    #dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=BatchSize)
    #report(model, criterion, dataloader, 'test')

if Cmd == 'test':
    # load data
    if len(DataFilename) != 1 or os.path.isfile(DataFilename[0]):
        print('--- Loading preprocessed data')
        data = []
        for filename in DataFilename:
            print('Loading %s' % (filename,))
            data.extend(pickle.load(open(filename, 'rb')))
    else:
        print('--- Preprocessing data')
        data = []
        for i in range(1, 21):
            filename_data = f'data/Session{i}/Session{i}.wav'
            filename_metadata = f'data/Session{i}/Session{i}.Table.1.selections.txt'
            metadata = pd.read_csv(filename_metadata, sep='\t').values
            metadata = metadata[:,3:5]
            data.extend(list(yield_samples(filename_data, metadata)))
        if DataFilename[0] != '':
            print('\n--- Saving preprocessed data')
            pickle.dump(data, open(DataFilename[0], 'wb'))

    # set up data loaders
    print('\n--- Loading data')
    dataset = Dataset(data, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BatchSize)

    num_data = len(dataset)
    num_pos = sum([torch.argmax(sample['target']) == 1 for sample in dataset])
    num_neg = num_data - num_pos
    print('Total samples: all=%d  pos/neg=%d/%d' % (len(dataset), num_pos, num_neg))

    # load model
    model = load_model(ModelFilename, device)
    model.eval()

    print('\n--- Testing report')
    report(model, criterion, dataloader)

if Cmd == 'eval':
    # load model
    model = load_model(ModelFilename, device)
    model.eval()

    # load data
    print('--- Preprocessing data')
    data = []
    if len(SoundFilename) == 0:
        print("ERROR: no input sound filenames specified with --sound")
        exit(1)
    for sound in SoundFilename:
        data.extend(list(yield_samples(sound)))

    # set up data loaders
    print('\n--- Loading data')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(data, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BatchSize)
    print('Total samples: %d' % (len(dataset),))

    print('\n--- Evaluating')
    n = 0
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            inputs = sample['input']
            positions = sample['position']

            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            for output, position in zip(outputs, positions):
                if torch.argmax(output) == 1:
                    print('Manatee call at %s—%s' % (dur2str((position[0]+position[1])/2.0),))
                    n = n + 1
    print('%d manatee calls found' % (n,))
