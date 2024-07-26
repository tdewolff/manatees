# Manatee audio classification
Detect manatee vocalisations recorded by hydrophones using the [audio spectrogram transformer model](https://github.com/YuanGongND/ast). This code repository accompanies our paper **Detection of manatee vocalisations using the Audio Spectrogram Transformer**, accepted at the *IEEE International Workshop on Machine Learning for Signal Processing (MLSP)* 2024. The project is a joint effort of Initiative for Data & AI (Universidad de Chile), AI4Climate (C-MINDS) and ECOSUR, with funding by Google.

## Usage
To pre-process the data set and train the model, run:
```bash
python manatee.py train
```

Note that your must have the Beauval dataset in the `data` directory.

Once a model has been trained, you can run using test mode or evaluation mode. Test mode uses pre-processed data to extract key performance metrics of the model (e.g. on a test data set), while evaluation mode will detect manatee vocalisations in a new audio file using the trained model (for example, `python manatee.py eval --sound soundfile.wav`).

See `python manatee.py --help` for more information:

```
usage: Manatee model training and evaluation [-h] [--epochs EPOCHS] [--split SPLIT] [--batch BATCH]
                                             [--lr LR] [--lr-step LR_STEP] [--lr-decay LR_DECAY]
                                             [--model MODEL] [--positive-split POSITIVE_SPLIT]
                                             [--data [DATA ...]] [--sound [SOUND ...]]
                                             [--report REPORT] [-n N]
                                             [{train,test,eval}]

positional arguments:
  {train,test,eval}

options:
  -h, --help            show this help message and exit
  --epochs=3            Number of epochs to run training
  --split=0.5           Training/validation split
  --batch=16            Batch size
  --lr=1e-6             Initial learning rate
  --lr-step=5           Learning rate scheduler epoch step
  --lr-decay=0.5        Learning rate scheduler step decay
  --model=model.pth     Output filename for trained model
  --positive-split=0.5  Percentage of positive samples (by adding negative samples)
  --data=[data.pkl]     Input filename for preprocessed data
  --sound=[]            Input filename for sound file for evaluation
  --report=report.pkl   Report filename
  -n=1                  Number of times to train
```
