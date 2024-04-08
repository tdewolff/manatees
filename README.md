# Manatee audio classification
Detect manatee vocalisations recorded by hydrophones using the [audio spectrogram transformer model](https://github.com/YuanGongND/ast). This code repository accompanies our paper submissions for ICASSP. The project is in collaboration with AI4Climate (C-MINDS) and ECOSUR, and is funded by Google.

## Usage
To pre-process the data set and train the model, run:
```bash
python manatee.py train
```

Note that your must have the Beauval data set in the `data` directory.

Once a model has been trained, you can run using test mode or evaluation mode. Test mode uses pre-processed data to extract key performance metrics of the model (e.g. on a test data set), while evaluation mode will detect manatee vocalisations in a new audio file using the trained model (for example, `python manatee.py eval --sound soundfile.wav`).

See `python manatee.py --help` for more information:

```
usage: Manatee model training and evaluation [-h] [--epochs EPOCHS] [--split SPLIT] [--batch BATCH]
                                             [--lr LR] [--lr-step LR_STEP] [--lr-decay LR_DECAY]
                                             [--model MODEL] [--pos-split POS_SPLIT]
                                             [--data [DATA ...]] [--sound [SOUND ...]]
                                             [--report REPORT]
                                             [{train,test,eval}]

positional arguments:
  {train,test,eval}

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs to run training
  --split SPLIT         Training/validation split
  --batch BATCH         Batch size
  --lr LR               Initial learning rate
  --lr-step LR_STEP     Learning rate scheduler epoch step
  --lr-decay LR_DECAY   Learning rate scheduler step decay
  --model MODEL         Output filename for trained model
  --positie-split POSITIVE_SPLIT
                        Percentage of positive samples (by adding negative samples)
  --data [DATA ...]     Input filename for preprocessed data
  --sound [SOUND ...]   Input filename for sound file for evaluation
  --report REPORT       Report filename
```
