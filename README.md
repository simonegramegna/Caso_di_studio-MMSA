# Caso di studio Metodi e modelli per la sicurezza delle applicazioni
Anomaly detection per rilevare le infezioni da Covid-19 usando modelli non supervisionati di anomaly detection

## 1. Linguaggio utilizzato
- Python 3.10.4 

## 2. Librerie richieste
- Numpy `pip install numpy`
- Pandas `pip install pandas`
- Sklearn `pip install scikit-learn`
- Matplotlib `pip install matplotlib`
- Tensorflow `pip install tensorflow`
- Seaborn `pip install seaborn`
- Pylab `pip install pylab-sdk`

## 3. Dataset
I dati utilizzati nell'esperimento sono presenti nella cartella */data* e sono due files:
1. ASFODQR_hr.csv
2. ASFODQR_steps.csv

Entrambi i dataset hanno 3 colonne relative a:
1. Codice utente: "user" - stringa (ASFODQR)
2. Data, ora, minuto della rilevazione: - data "datetime"
3. Valore associato: - intero "heartrate" nel file ASFODQR_hr.csv e "steps" nel file ASFODQR_steps.csv

## 4. Esecuzione dell'esperimento LAAD 
Andare nella cartella */laad_ASFODQR* è presente lo script **laad_covid19.py**, eseguirlo il comando:
`python laad_covid19.py  --heart_rate ASFODQR_hr.csv --steps ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14`
I risultati una volta terminata l'esecuzione dello script sono visibili nella stessa cartella.

## 5. Esecuzione dell'esperimento 
Nella cartella principale è presente lo script **rhrad_offline.py**, eseguirlo mediante il comando:
`python rhrad_offline.py --heart_rate data/ASFODQR_hr.csv --steps data/ASFODQR_steps.csv --myphd_id asfodqr --anomalies ASFODQR_exp.csv --symptom_date 2024-08-07 --diagnosis_date 2024-08-14 --outliers_fraction 0.25 --random_seed 10`
I risultati sono stampati a video inoltre le matriche ed i grafici ottenuti dall'esperimento si trovano nella cartella */reports* in cui ci sono:
- Grafici delle anomalie rilevate da ciascuno dei quattro modelli
- File csv delle anomalie rilevate per ciascuno dei modelli
- Matrici di confusione

## 6. Repository utilizzate
1. Esperimento LAAD, autoencoder con cui ho ottenuto i risultati di riferimento [LAAD-Github](https://github.com/gireeshkbogu/LAAD)
2. Esperimento utilizzato per fare il confronto tra i modelli non supervisionati [Anomaly detection COVID](https://github.com/gireeshkbogu/AnomalyDetect)
