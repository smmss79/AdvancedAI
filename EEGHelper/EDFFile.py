from pyedflib import highlevel
import numpy as np

# write an edf file
def write_file(ch_names,sampling_freq,subject_name,subject_gender,file_name,signal_data=[]):
    
    if signal_data == []:   
        signals = np.random.rand(5, 256*300)*200 # 300 seconds/ 256 Hz / 5 channels of random signal
    else:
        signals = signal_data

    # channel_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5']
    channel_names = ch_names
    # signal_headers = highlevel.make_signal_headers(channel_names, sample_frequency=256)
    signal_headers = highlevel.make_signal_headers(channel_names, sample_frequency=sampling_freq)
    # header = highlevel.make_header(patientname='patient_x', gender='Female')
    header = highlevel.make_header(patientname=subject_name, gender=subject_gender)
    # highlevel.write_edf('edf_file.edf', signals, signal_headers, header)
    highlevel.write_edf(file_name+".edf", signals, signal_headers, header)


# read an edf file
def read_file(file_name,channel_names):
    # signals, signal_headers, header = highlevel.read_edf('edf_file.edf', ch_names=['ch1', 'ch2'])
    signals, signal_headers, header = highlevel.read_edf(file_name, ch_names=channel_names)
    # print('first element: ',signal_headers[0]['sample_frequency']) # prints for default signal 256

    return (signals, signal_headers, header)

# get a list of individual records directory from RECORDS file in the dataset folder
def get_records_directory( RECORDS_file_path,subject_count, record_count):

    RECORDS_f = open(RECORDS_file_path,'r')
    record_file_directory = []
    for i in range(subject_count):
        for i in range(record_count):
            record_file_directory.append(RECORDS_f.readline()[:-1])

    return record_file_directory



# create a list  ['ch0', 'ch1' , ....]
def create_channel_name_list(self,channel_number):

    temp = []
    for i in range(channel_number):
        temp.append('ch'+i)
    
    return temp



# Load data from dataset to a list object
def load_records_to_list(sample_subject_list,record_count,record_file_directory,channel_name):

    records_data = []

    for subject in sample_subject_list:
        # records_data.append([]) 
        for record in range(record_count):
            temp_record_file_directory = "D:\python projects\EEG_ML\Dataset\\" + record_file_directory[subject*record_count+record]
            records_data.append(read_file(temp_record_file_directory, channel_name))
        
    return records_data

# seperate sinal to T0/T1/T2 
# each has len(records_data) list correspond to each record
def seperate_by_labels(records_data,sampling_frequency):

    signal_T0, signal_T1, signal_T2 = [], [], []


    for rec in range(len(records_data)):

        signal_T0.append([])
        signal_T1.append([])
        signal_T2.append([])

        for item in records_data[rec][2]['annotations']:

            start_index = int(item[0] * sampling_frequency)
            end_index = int( start_index + item[1] * sampling_frequency)
            temp =[]
            for channels in range(64):

                temp.append(records_data[rec][0][channels][start_index:end_index])

            if (len(temp)>0):

                if item[2] == 'T0':    

                    signal_T0[rec].append(temp)

                elif item[2] == 'T1':
                    
                    signal_T1[rec].append(temp)

                elif item[2] == 'T2':

                    signal_T2[rec].append(temp)

    return [signal_T0, signal_T1, signal_T2]


   



     
        
    # # drop a channel from the file or anonymize edf
    # highlevel.drop_channels('edf_file.edf', to_drop=['ch2', 'ch4'])
    # highlevel.anonymize_edf('edf_file.edf', new_file='anonymized.edf'
    #                             to_remove=['patientname', 'birthdate'],
    #                             new_values=['anonymized', ''])
    # # check if the two files have the same content
    # highlevel.compare_edf('edf_file.edf', 'anonymized.edf')
    # # change polarity of certain channels
    # highlevel.change_polarity('file.edf', channels=[1,3])
    # # rename channels within a file
    # highlevel.rename_channels('file.edf', mapping={'C3-M1':'C3'})
