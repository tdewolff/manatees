# Manatee audio classification
Detect manatee vocalisations recorded by hydrophones using the [audio spectrogram transformer model](https://github.com/YuanGongND/ast). This code repository accompanies our paper submissions for ICASSP. The project is in collaboration with AI4Climate (C-MINDS) and ECOSUR, and is funded by Google.

## Usage
To pre-process the data set and train the model, run:
```bash
python manatee.py train
```

Note that your must have the Beauval data set in the `data` directory.

Once a model has been trained, you can run using test mode or evaluation mode. Test mode uses pre-processed data to extract key performance metrics of the model (e.g. on a test data set), while evaluation mode will detect manatee vocalisations in a new audio file using the trained model (for example, `python manatee.py eval soundfile.wav`).

See `python manatee.py --help` for more information.
