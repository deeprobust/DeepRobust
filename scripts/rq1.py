import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import scipy.stats
'''
Python3 script

RQ1 histograms and tables
Step1: please set data_path variable to the absolute path of the folder holding all the neighbour accuracy data.
Step2: make sure there are three folders under data_path directory, 
       the three folders should be "cifar10", "fmnist" and "svhn".
Step3: Inside each of the three folders, there should be 6 neighbour data files. 
       The 6 files are corresponding to train data and test data for each of the three models.
Step4: Run this scripts. It will generate pdfs and output correlations for all histograms and tables in RQ1.
'''
data_path = "/home/user/deeprobust/data/"
dataset = ["cifar10", "fmnist", "svhn"]
d_dataset = ["CIFAR-10", "F-MNIST", "SVHN"]
trainprefix = "/natural_train_embed_and_neighbor_acc_"
trainpostfix = "_random_rotate_and_shift_50_x4.npz"
testprefix = "/natural_test_embed_and_neighbor_acc_"
testpostfix = "_random_rotate_and_shift_x4.npz"
neighbourkey = ["train_neighbor_avg_acc", "test_neighbor_avg_acc"]
models = ["vgg16", "resnet", "wrn"]
d_models = ["VGG", "ResNet","WRN"]
prefix = [trainprefix, testprefix]
postfix = [trainpostfix, testpostfix]

import matplotlib.pyplot as plt



'''
table 3 in paper
'''
def get_correlation_table3():
    print("table 3 in paper")
    for d in dataset:
        print(d)
        number = []
        for mi in range(len(models)):
            for mode in [0, 1]:
                m = models[mi]
                raw_data = np.load(data_path + d + prefix[mode] + m + postfix[mode])
                neighbour_accuracy = raw_data[neighbourkey[mode]]
                data_to_plot = []
                for i in range(len(neighbour_accuracy)):
                    data_to_plot.append(neighbour_accuracy[i])

                (n, bins, patches) = plt.hist(data_to_plot, color='grey', edgecolor='black',
                                             bins=int(20))
                number.append(n)
        print("train")
        print(models[0] + " vs " + models[1])
        print(scipy.stats.spearmanr(number[0], number[2])[0])
        print(scipy.stats.pearsonr(number[0], number[2])[0])
        print(scipy.stats.pearsonr(number[0][:-1], number[2][:-1])[0])
        print(models[0] + " vs " + models[2])
        print(scipy.stats.spearmanr(number[0], number[4])[0])
        print(scipy.stats.pearsonr(number[0], number[4])[0])
        print(scipy.stats.pearsonr(number[0][:-1], number[4][:-1])[0])
        print(models[1] + " vs " + models[2])
        print(scipy.stats.spearmanr(number[2], number[4])[0])
        print(scipy.stats.pearsonr(number[2], number[4])[0])
        print(scipy.stats.pearsonr(number[2][:-1], number[4][:-1])[0])
        print("test")
        print(models[0] + " vs " + models[1])
        print(scipy.stats.spearmanr(number[1], number[3])[0])
        print(scipy.stats.pearsonr(number[1], number[3])[0])
        print(scipy.stats.pearsonr(number[1][:-1], number[3][:-1])[0])
        print(models[0] + " vs " + models[2])
        print(scipy.stats.spearmanr(number[1], number[5])[0])
        print(scipy.stats.pearsonr(number[1], number[5])[0])
        print(scipy.stats.pearsonr(number[1][:-1], number[5][:-1])[0])
        print(models[1] + " vs " + models[2])
        print(scipy.stats.spearmanr(number[3], number[5])[0])
        print(scipy.stats.pearsonr(number[3], number[5])[0])
        print(scipy.stats.pearsonr(number[3][:-1], number[5][:-1])[0])

'''
table 2 in paper
'''
def get_correlation_table2():
    print("table 2 in paper")
    for d in dataset:
        print(d)
        number = []
        for mi in range(len(models)):
            for mode in [0, 1]:
                m = models[mi]
                raw_data = np.load(data_path + d + prefix[mode] + m + postfix[mode])
                neighbour_accuracy = raw_data[neighbourkey[mode]]
                data_to_plot = []
                for i in range(len(neighbour_accuracy)):
                    data_to_plot.append(neighbour_accuracy[i])

                (n, bins, patches) = plt.hist(data_to_plot, color='grey', edgecolor='black',
                                             bins=int(20))
                number.append(n)
        print(models[0])
        print(scipy.stats.spearmanr(number[0], number[1])[0])
        print(scipy.stats.pearsonr(number[0], number[1])[0])
        print(scipy.stats.pearsonr(number[0][:-1], number[1][:-1])[0])
        print(models[1])
        print(scipy.stats.spearmanr(number[2], number[3])[0])
        print(scipy.stats.pearsonr(number[2], number[3])[0])
        print(scipy.stats.pearsonr(number[2][:-1], number[3][:-1])[0])
        print(models[2])
        print(scipy.stats.spearmanr(number[4], number[5])[0])
        print(scipy.stats.pearsonr(number[4], number[5])[0])
        print(scipy.stats.pearsonr(number[4][:-1], number[5][:-1])[0])


'''
figure 6 in paper
'''
def get_neighbour_accuracy_change_hist_testing_only():
    fig, axs = plt.subplots(1, 3, figsize=(30, 10), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    total_count = 0
    total_data = 0
    for di in range(len(dataset)):
        for mode in [1]:
            d = dataset[di]
            # m1,m2 can be changed to any two of the three models: models[0],model[1],models[2]
            # currently it is vgg16 vs resnet.
            # "models" is a global variable defined at the top of this script.
            m1 = models[0]
            m2 = models[1]

            ind = mode * 3 + di
            ax = axs[di]
            raw_data = np.load(data_path + d + prefix[mode] + m1 + postfix[mode])
            raw_data2 = np.load(data_path + d + prefix[mode] + m2 + postfix[mode])


            data_to_plot = []
            data_to_plot += list(raw_data[neighbourkey[mode]] - raw_data2[neighbourkey[mode]])
            count = 0
            for dp in data_to_plot:
                if dp < 0.2 and dp > -0.2:
                    count += 1
                    total_count += 1
            print("dataset: "+d)
            if mode == 0:
                print("per-traindata neighbour accuracy change between -0.2 to 0.2")
            else:
                print("per-testdata neighbour accuracy change between -0.2 to 0.2")
            print(m1 + " vs " + m2)
            print("percentage of data: " + str(count/len(raw_data[neighbourkey[mode]])))
            total_data += len(raw_data[neighbourkey[mode]])
            (n, bins, patches) = ax.hist(data_to_plot, color='grey', edgecolor='black',
                     bins=int(20))

            ax.tick_params(axis='x', labelsize=45)
            ax.tick_params(axis='y', labelsize=45)
            ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])

            if mode == 1:
                if di == 1:
                    ax.set_xlabel("Neighbour Accuracy Difference\n" + d_dataset[di], fontsize=60)
                else:
                    ax.set_xlabel("\n" + d_dataset[di], fontsize=60)

            else:
                ax.set_xlabel('')

            if ind == 0:
                ax.set_ylabel('#Training Data', fontsize=60)
            elif ind == 3:
                ax.set_ylabel('#Testing Data', fontsize=60)
            else:
                ax.set_ylabel('')

    for ax in axs.flat:
        ax.label_outer()
    plt.savefig("diff.pdf", bbox_inches='tight', dpi=1000)
    print("figure 6 in paper is saved in 'diff.pdf'")
    print(total_count/total_data)


def get_histogram_combine_only_testing(dataset="cifar10"):
    fig, axs = plt.subplots(1, 3, figsize=(30, 10), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0.05})
    xlabeltext = ['Train Data Neighbour Accuracy', 'Test Data Neighbour Accuracy']

    d = dataset
    for mi in range(len(models)):
        for mode in [1]:
            ax = axs[mi]
            m = models[mi]
            ind = mode*3 + mi
            raw_data = np.load(data_path + d + prefix[mode] + m + postfix[mode])
            neighbour_accuracy = raw_data[neighbourkey[mode]]
            data_to_plot = []
            for i in range(len(neighbour_accuracy)):
                data_to_plot.append(neighbour_accuracy[i])

            (n, bins, patches) = ax.hist(data_to_plot, color='grey', edgecolor='black',
                                          bins=int(20))

            ax.tick_params(axis='x', labelsize=45)
            ax.tick_params(axis='y', labelsize=45)

            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

            if mode == 1:
                if mi == 1:
                    ax.set_xlabel("Neighbour Accuracy\n" + d_models[mi], fontsize=60)
                else:
                    ax.set_xlabel("\n" + d_models[mi], fontsize=60)
            else:
                ax.set_xlabel('')

            if ind == 0:
                ax.set_ylabel('#Training Data', fontsize=60)
            elif ind == 3:
                ax.set_ylabel('#Testing Data', fontsize=60)
            else:
                ax.set_ylabel('')

            ax2 = ax.inset_axes([0.15, 0.3, 0.6, 0.6])
            (n, bins, patches) = ax2.hist(data_to_plot, color='grey', edgecolor='black',
                                         bins=int(16), range=(min(data_to_plot), 0.80))
            #ax2.set_xticks([])
            #ax2.set_yticks([])
            ax2.tick_params(axis='x', labelsize=30)
            ax2.set_xticks([0, 0.8])
            ax2.tick_params(axis='y', labelsize=30)
            #ax.indicate_inset_zoom(ax2, linewidth=10)
            mark_inset(ax, ax2, loc1=1, loc2=2, fc="none", ec="0.5", ls="--", lw=5)
    for ax in axs.flat:
        ax.label_outer()
    #plt.show()
    plt.savefig(d + ".pdf", bbox_inches='tight', dpi = 1000)
    print("figure 5 in paper is saved in '<dataset_name>.pdf'")


if __name__ == '__main__':
    print("--------------------------------------------------")
    get_histogram_combine_only_testing("cifar10") #figure 5 in rq1. the arguments can be "fmnist", "cifar10" or "svhn".
    print("--------------------------------------------------")
    get_neighbour_accuracy_change_hist_testing_only() #figure 6 in rq1.
    print("--------------------------------------------------")
    get_correlation_table2() #table2 in rq1
    print("--------------------------------------------------")
    get_correlation_table3() #table3 in rq1
    print("--------------------------------------------------")