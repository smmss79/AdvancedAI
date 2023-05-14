import matplotlib.pyplot as plt



def plot_one_subject(figsize_tuple, fig_dpi, isPlotNotShow,records_data,ch_start,sampling_frequency, ch_stop,save_dpi=50):

    # print(len(records_data[5][0][0]))

    plt.figure(figsize=figsize_tuple, dpi=fig_dpi, )
    for task in range(len(records_data)):
        
        # print(len(records_data[0][0]))

        

        if (isPlotNotShow):
            plt.subplot(14,1,task+1)
            draw_annotations(records_data[task][2]['annotations'],sampling_frequency)
            for i in range(ch_start,ch_stop):
                plt.plot(records_data[task][0][i])
        else:
            draw_annotations(records_data[task][2]['annotations'],sampling_frequency)
            for i in range(ch_start,ch_stop):
                plt.plot(records_data[task][0][i])
            plt.savefig('my_plot'+str(task)+'.png',dpi=save_dpi)
            plt.cla()


    handles, lbls = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(lbls,handles))
    plt.legend(by_label.values(),by_label.keys(),loc = 'upper left', bbox_to_anchor =(1,1))
    plt.show()  


def draw_annotations(annotations,sampling_freq):

    for item in annotations:

        if item[2] == 'T0':
            plt.axvline(x= sampling_freq*item[0], color='r',label = item[2] )  
        elif item[2] == 'T1':
            plt.axvline(x= sampling_freq*item[0], color='g', label = item[2])  
        elif item[2] == 'T2':
            plt.axvline(x= sampling_freq*item[0], color='y',label = item[2] )
          


