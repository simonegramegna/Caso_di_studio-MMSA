import warnings
warnings.filterwarnings('ignore')
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, confusion_matrix, f1_score, recall_score, ConfusionMatrixDisplay
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

####################################

parser = argparse.ArgumentParser(description='Find anomalies in wearables time-series data.')
parser.add_argument('--heart_rate', metavar='', help ='raw heart rate count with a header = heartrate')
parser.add_argument('--steps',metavar='', help ='raw steps count with a header = steps')
parser.add_argument('--myphd_id',metavar='', default = 'myphd_id', help ='user myphd_id')
parser.add_argument('--figure', metavar='',  default = 'myphd_id_anomalies.pdf', help='save predicted anomalies as a PDF file')
parser.add_argument('--anomalies', metavar='', default = 'myphd_id_anomalies.csv', help='save predicted anomalies as a CSV file')
parser.add_argument('--symptom_date', metavar='', default = 'NaN', help = 'symptom date with y-m-d format')
parser.add_argument('--diagnosis_date', metavar='', default = 'NaN',  help='diagnosis date with y-m-d format')
parser.add_argument('--outliers_fraction', metavar='', type=float, default=0.1, help='fraction of outliers or anomalies')
parser.add_argument('--random_seed', metavar='', type=int, default=10, help='random seed')
args = parser.parse_args()

# as arguments
fitbit_oldProtocol_hr = args.heart_rate
fitbit_oldProtocol_steps = args.steps
myphd_id = args.myphd_id
myphd_id_figure = myphd_id +"___"+ args.figure
myphd_id_anomalies = args.anomalies
symptom_date = args.symptom_date
diagnosis_date = args.diagnosis_date
RANDOM_SEED = args.random_seed
outliers_fraction =  args.outliers_fraction

###################################
# Directory in cui sono salvati i report dei modelli
dir_reports = "reports/"

###################################

class RHRAD_offline:

   # Resting Heart Rate (RHR) ------------------------------------------------------

    def RHR(self, heartrate, steps):
        """
        This function uses heart rate and steps data to infer restign heart rate.
        It filters the heart rate with steps that are zero and also 12 minutes ahead.
        """
        # heart rate data
        df_hr = pd.read_csv(heartrate)
        df_hr = df_hr.set_index('datetime')
        df_hr.index.name = None
        df_hr.index = pd.to_datetime(df_hr.index)

        # steps data
        df_steps = pd.read_csv(steps)
        df_steps = df_steps.set_index('datetime')
        df_steps.index.name = None
        df_steps.index = pd.to_datetime(df_steps.index)

        df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
        df1 = df1.resample('1min').mean()
        df1 = df1.dropna()

        # define RHR as the HR measurements recorded when there were zero steps taken during 
        # a rolling time window of the preceding 12 minutes (including the current minute).
        df1['steps_window_12'] = df1['steps'].rolling(12).sum()
        df1 = df1.loc[(df1['steps_window_12'] == 0)]
        return df1

    # pre-processing ------------------------------------------------------
    def pre_processing(self, df1):
        """
        It takes resting heart rate data and applies moving averages to smooth the data and 
        aggregates to one hour by taking the avegare values
        """

        # smooth data
        df_nonas = df1.dropna()
        df1_rom = df_nonas.rolling(400).mean()

        # resample
        df1_resmp = df1_rom.resample('1H').mean()
        df2 = df1_resmp.drop(['steps'], axis=1)
        df2 = df2.drop(['steps_window_12'], axis=1)
        df2 = df2.dropna()

        return df2

    # standardization ------------------------------------------------------
    def standardization(self, data_exp):
        """
        Standardize the data with zero meann and unit variance (Z-score).
        """
        data_scaled = StandardScaler().fit_transform(data_exp.values)
        data_scaled_features = pd.DataFrame(
            data_scaled, index=data_exp.index, columns=data_exp.columns)

        data_df = pd.DataFrame(data_scaled_features)
        data = pd.DataFrame(data_df).fillna(0)

        return data

    # train model and predict anomalies -----------------------------------
    def anomaly_detection(self, standardized_data, model):
        """
        This function takes the standardized data and detects outliers using Gaussian density estimation.
        """
        model.fit(standardized_data)
        preds = pd.DataFrame(model.predict(standardized_data))
        preds = preds.rename(lambda x: 'anomaly' if x == 0 else x, axis=1)
        data = standardized_data.reset_index()
        data = data.join(preds)
        
        data.loc[data['anomaly'] == -1, 'anomaly'] = 0
       
        return data

    # Visualization ------------------------------------------------------
    def visualize(self, data, symptom_date, diagnosis_date, start_file: str):
        """
        visualize results and also save them to a .csv file 

        """
        try:

            with plt.style.context('fivethirtyeight'):
                fig, ax = plt.subplots(1, figsize=(80,15))
                a = data.loc[data['anomaly'] == 0, ('index', 'heartrate')] #anomaly
                b = a[(a['heartrate']> 0)]
                ax.bar(data['index'], data['heartrate'], linestyle='-',color='midnightblue' ,lw=6, width=0.01)
                ax.scatter(b['index'],b['heartrate'], color='red', label='Anomaly', s=500)
                
                ax.tick_params(axis='both', which='major', color='blue', labelsize=60)
                ax.tick_params(axis='both', which='minor', color='blue', labelsize=60)
                ax.set_title(myphd_id,fontweight="bold", size=50) # Title
                ax.set_ylabel('Std. HROS\n', fontsize = 50) # Y label
                ax.axvline(pd.to_datetime(symptom_date), color='red', zorder=1, linestyle='--', lw=8) # Symptom date 
                ax.axvline(pd.to_datetime(diagnosis_date), color='purple',zorder=1, linestyle='--', lw=8) # Diagnosis date
                ax.tick_params(axis='both', which='major', labelsize=60)
                ax.tick_params(axis='both', which='minor', labelsize=60)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                
                ax.grid(zorder=0)
                ax.grid(True)
               
                plt.xticks(fontsize=30, rotation=90)
                plt.yticks(fontsize=50)
                ax.patch.set_facecolor('white')
                fig.patch.set_facecolor('white')

                figure = fig.savefig(dir_reports+start_file+"_"+myphd_id_figure, bbox_inches='tight') 

                # Anomaly results
                b['Anomalies'] = myphd_id
                b.to_csv(dir_reports+start_file+"_"+myphd_id_anomalies,
                         mode='a', header=False)
                return figure

        except:
            with plt.style.context('fivethirtyeight'):
                fig, ax = plt.subplots(1, figsize=(80,15))
                a = data.loc[data['anomaly'] == 0, ('index', 'heartrate')] #anomaly
                b = a[(a['heartrate']> 0)]
                ax.bar(data['index'], data['heartrate'], linestyle='-',color='midnightblue' ,lw=6, width=0.01)
                ax.scatter(b['index'],b['heartrate'], color='red', label='Anomaly', s=1000)
                ax.tick_params(axis='both', which='major', color='blue', labelsize=60)
                ax.tick_params(axis='both', which='minor', color='blue', labelsize=60)
                ax.set_title(myphd_id,fontweight="bold", size=50) # Title
                ax.set_ylabel('Std. HROS\n', fontsize = 50) # Y label
               
                ax.tick_params(axis='both', which='major', labelsize=60)
                ax.tick_params(axis='both', which='minor', labelsize=60)
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                ax.grid(zorder=0)
                ax.grid(True)
                plt.xticks(fontsize=30, rotation=90)
                plt.yticks(fontsize=50)
                ax.patch.set_facecolor('white')
                fig.patch.set_facecolor('white')     
                figure = fig.savefig(myphd_id_figure, bbox_inches='tight')

                # Anomaly results
                b['Anomalies'] = myphd_id
                b.to_csv(dir_reports+start_file+"_" +
                         myphd_id_anomalies, mode='a', header=False)
                return figure

def model_performance(data: pd.DataFrame, anomaly_truth: str, image: str, title_graph: str):

    # analisi delle performance rispetto al modello di riferimento
    truth_df = pd.read_csv(anomaly_truth)

    truth_df["anomaly"] = truth_df["anomaly"].map({True: -1, False: 1})

    truth_df_dict = truth_df.to_dict()
    data_dict = data.to_dict()

    i = 0
    time_str = {}
    for k in data_dict['index'].values():
        time_str[i] = k.strftime("%Y-%m-%d %H:%M:%S")
        i = i + 1

    test_rhr_dict = {}
    test_rhr_dict["timestamp"] = time_str
    test_rhr_dict["RHR"] = data_dict["heartrate"]
    test_rhr_dict["anomaly_predicted"] = data_dict["anomaly"]
    test_rhr_dict["anomaly_truth"] = {}

    i = 0
    for date_str in test_rhr_dict["timestamp"].values():
        val_anomaly = 1

        if date_str in truth_df_dict["index"].values():
            val_anomaly = 0

        test_rhr_dict["anomaly_truth"][i] = val_anomaly
        i = i + 1

    test_rhr_df = pd.DataFrame.from_dict(test_rhr_dict)

    # calcolo delle metriche di accuratezza, precision, recall, f1_score ed f1_beta
    accuracy = accuracy_score(
        test_rhr_df["anomaly_truth"].values, test_rhr_df["anomaly_predicted"].values)

    precision = precision_score(
        test_rhr_df["anomaly_truth"].values, test_rhr_df["anomaly_predicted"].values)

    recall = recall_score(
        test_rhr_df["anomaly_truth"].values, test_rhr_df["anomaly_predicted"].values)

    f1_metric = f1_score(test_rhr_df["anomaly_truth"].values,
                  test_rhr_df["anomaly_predicted"].values)
    
    fbeta_metric = fbeta_score(
        test_rhr_df["anomaly_predicted"].values, test_rhr_df["anomaly_truth"].values, beta=0.5)

    # matrice di confusione
    conf_matrix = confusion_matrix(
        test_rhr_df["anomaly_truth"].values, test_rhr_df["anomaly_predicted"].values)

    disp_matrix = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp_matrix.plot()
    plt.title(title_graph)
    plt.grid(False)
    plt.savefig(dir_reports+image)
    
    return accuracy, precision, recall, f1_metric, fbeta_metric


def experiment_EllipticEnvelope(anomaly_truth: str):
    model = RHRAD_offline()

    df1 = model.RHR(fitbit_oldProtocol_hr, fitbit_oldProtocol_steps)
    df2 = model.pre_processing(df1)
    std_data = model.standardization(df2)

    model_EllipticEnvelope = EllipticEnvelope(
        contamination=outliers_fraction, random_state=RANDOM_SEED, support_fraction=0.7)

    data = model.anomaly_detection(std_data, model_EllipticEnvelope)
    model.visualize(data, symptom_date, diagnosis_date, "EllipticEnvelope")
    
    conf_matrix_file = "confusion_matrix_elliptic.png"
    title_confusion_matrix = "Matrice di confusione EllipticEnvelope"

    accuracy, precision, recall, f1_metric, fbeta_metric = model_performance(
        data, anomaly_truth, conf_matrix_file, title_confusion_matrix)

    print("Metriche calcolate per l'EllipticEnvelope\n")
    print("Accuracy : " + str(accuracy))
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    print("F1 score : " + str(f1_metric))
    print("F-Beta score : " + str(fbeta_metric))
    print("\n")


def experiment_IsolationForest(anomaly_truth: str):
    model = RHRAD_offline()

    df1 = model.RHR(fitbit_oldProtocol_hr, fitbit_oldProtocol_steps)
    df2 = model.pre_processing(df1)
    std_data = model.standardization(df2)

    model_IsolationForest = IsolationForest(contamination=outliers_fraction)

    data = model.anomaly_detection(std_data, model_IsolationForest)
    model.visualize(data, symptom_date, diagnosis_date, "IsolationForest")

    conf_matrix_file = "confusion_matrix_IsolationForest.png"
    title_confusion_matrix = "Matrice di confusione IsolationForest"

    accuracy, precision, recall, f1_metric, fbeta_metric = model_performance(
        data, anomaly_truth, conf_matrix_file, title_confusion_matrix)

    print("Metriche calcolate per l'IsolationForest\n")
    print("Accuracy : " + str(accuracy))
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    print("F1 score : " + str(f1_metric))
    print("F-Beta score : " + str(fbeta_metric))
    print("\n")


def experiment_LocalOutlierFactor(anomaly_truth: str):
    model = RHRAD_offline()

    df1 = model.RHR(fitbit_oldProtocol_hr, fitbit_oldProtocol_steps)
    df2 = model.pre_processing(df1)
    std_data = model.standardization(df2)

    model_LocalOutlierFactor = LocalOutlierFactor(novelty=True, contamination=outliers_fraction)

    data = model.anomaly_detection(std_data, model_LocalOutlierFactor)
    model.visualize(data, symptom_date, diagnosis_date, "LocalOutlierFactor")

    conf_matrix_file = "confusion_matrix_LocalOutlierFactor.png"
    title_confusion_matrix = "Matrice di confusione LocalOutlierFactor"

    accuracy, precision, recall, f1_metric, fbeta_metric = model_performance(
        data, anomaly_truth, conf_matrix_file, title_confusion_matrix)

    print("Metriche calcolate per il LocalOutlierFactor\n")
    print("Accuracy : " + str(accuracy))
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    print("F1 score : " + str(f1_metric))
    print("F-Beta score : " + str(fbeta_metric))
    print("\n")


def experiment_OneClassSVM(anomaly_truth: str):
    model = RHRAD_offline()

    df1 = model.RHR(fitbit_oldProtocol_hr, fitbit_oldProtocol_steps)
    df2 = model.pre_processing(df1)
    std_data = model.standardization(df2)

    outliers_fraction = 0.25
    model_OneClassSVM = OneClassSVM(nu=0.95*outliers_fraction)

    data = model.anomaly_detection(std_data, model_OneClassSVM)
    model.visualize(data, symptom_date, diagnosis_date, "OneClassSVM")

    conf_matrix_file = "confusion_matrix_OneClassSVM.png"
    title_confusion_matrix = "Matrice di confusione OneClassSVM"

    accuracy, precision, recall, f1_metric, fbeta_metric = model_performance(
        data, anomaly_truth, conf_matrix_file, title_confusion_matrix)

    print("Metriche calcolate per il OneClassSVM\n")
    print("Accuracy : " + str(accuracy))
    print("Precision : " + str(precision))
    print("Recall : " + str(recall))
    print("F1 score : " + str(f1_metric))
    print("F-Beta score : " + str(fbeta_metric))
    print("\n")

if __name__ == '__main__':
    anomaly_truth_file = "laad_ASFODQR/ASFODQR_anomalies.csv"

    print("\n\n\n")

    # Esperimento con EllipticEnvelope
    experiment_EllipticEnvelope(anomaly_truth_file)

   
    # Esperimento con IsolationForest
    experiment_IsolationForest(anomaly_truth_file)

    # esperimento con LocalOutlierFactor
    experiment_LocalOutlierFactor(anomaly_truth_file)
    
    # esperimento con OneClassSVM
    experiment_OneClassSVM(anomaly_truth_file)
    