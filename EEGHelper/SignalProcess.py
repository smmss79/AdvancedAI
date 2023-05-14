import numpy as np
from scipy.fftpack import fft


# recive data of single channel and create features via window function
def sliding_window(data,window_length):


    window_start = 0
    window_end = window_length

    features = []
    window_count = 0

    while(window_end<len(data)-1):
        
        # print(window_count)
        
        window_data = data[window_start:window_end]
        
        
        if(len(window_data)>0):

            # print("INNER!!!")

            features.append([])
            # here you must extract features
           
            # time domain
            features[window_count].append(np.mean(window_data))
            features[window_count].append(np.var(window_data))
            features[window_count].append(np.max(window_data))
            features[window_count].append(np.min(window_data))
            # features[window_count].append(np.(window_data))

            # frequency domain
            fft_temp = np.abs( fft(window_data) )
            fft_max = np.max( fft_temp )
            fft_max_ind = list(fft_temp).index(fft_max)

            features[window_count].append(np.mean( fft_temp ) )
            features[window_count].append(np.var( fft_temp ) )
            features[window_count].append( fft_max )
            features[window_count].append( fft_max_ind )
            # features[window_count].append(  )
        
            # end feature exrtract

        # print(features)

        window_count += 1
        window_end = min(window_end+window_length, len(data)-1)
        window_start = max (window_start+window_length, window_end-window_length)

    return features



def extract_features(records_datas,task_count=109,window_length=160,channel_list=[54,55,56]):

# it takes 19085271100 nano seconds (19.08s) with window_size = 80 
# and 12070455000 nano seconds (12.07s) with window_size = 128
# and 9698057700 nano seconds (9.69s) with window_size = 160 for 64 channels 14 tasks
# takes 222117400 nano seconds (0.22s) with window_size = 128, 32 channels and 1 task 
# takes 471383100 nano seconds (0.47s) with window_size = 128, all channels and 1 task 
# takes 906229200 nano seconds (0.91s) with window_size = 64, all channels and 1 task 

    temp = []
    for record in range(task_count):
        temp.append([])
        for channel_num in channel_list:
            channel_data = records_datas[record][0][channel_num]
            # print(len(channel_data))
            temp[record].append( sliding_window(channel_data, window_length))

            # print(temp)

    return temp




def extract_features_labelled(labelled_records_datas,task_list=[1,3],window_length=160,channel_list=[54,55,56]):

    temp = []
    temp2 = []

    for label in range(len(labelled_records_datas)):
        # temp.append([])
        for record in task_list:
            # temp[label].append([])
            for repetation in range(len(labelled_records_datas[label][record])):
                for channel_num in channel_list:
                    channel_data = labelled_records_datas[label][record][repetation][channel_num]
                    
                    for item in sliding_window(channel_data, window_length):
                        temp.append(item)
                        temp2.append(label)
                    # temp[label][record].append(sliding_window(channel_data, window_length))
                    # temp[label][record].append(label)

                    # print(temp[-1][-1])
    return temp,temp2

# def extract_features_labelled(labelled_records_datas,task_count=109,window_length=160,channel_list=[54,55,56]):

#     temp = []
#     for label in range(len(labelled_records_datas)):
#         temp.append([])
#         for record in range(task_count):
#             temp[label].append([])
#             for repetation in range(len(labelled_records_datas[label][record])):
#                 for channel_num in channel_list:
#                     channel_data = labelled_records_datas[label][record][repetation][channel_num]
#                     # print(len(channel_data))
#                     temp[label][record].append(sliding_window(channel_data, window_length))
#                     temp[label][record].append(label)

#                     # print(temp)
#     return temp