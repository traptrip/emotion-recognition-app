# Emotion recognition app & training scripts
Python 3.9.12

# Data
**Image data**
- [FER2013](https://www.kaggle.com/competitions/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
- [CK+](https://www.kaggle.com/datasets/shawon10/ckplus)

**Audio data**
- [RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
- [SAVEE](https://www.kaggle.com/datasets/barelydedicated/savee-database)

# Train Model
1. Change global variable CFG_NAME in run_train.py to appropriate training config name
2. Run training script
```bash
python run_train.py
```

# Run application (need to have cuda & nvidia-docker available)
1) `bash docker-compose up -d --build`
2) Go to http://localhost:8004/

