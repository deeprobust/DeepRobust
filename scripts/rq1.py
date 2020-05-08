import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.stats
'''
Python3 script

This analysis script is based on the
'data generated from the Extract feature vectors for neighbors and train/test a detector'
subsection in the readme.md.

RQ1 histograms and tables
Step1: please set data_path variable to the path of the folder holding all the neighbour accuracy data ( the default should work).
Step2: make sure there are three folders under data_path directory,
       the three folders should be "cifar10", "fmnist" and "svhn".
Step3: Inside each of the three folders, there should be 6 neighbour data files.
       The 6 files are corresponding to train data and test data for each of the three models.
Step4: Run this scripts. It will generate pdfs and output correlations for all histograms and tables in RQ1.
'''
data_path = "tmp_data_without_neighbor/"
dataset = ["cifar10", "fmnist", "svhn"]

trainprefix = "/natural_train_embed_and_neighbor_acc_"
trainpostfix = "_sweeping_rotate_and_random_shift_x4.npz"
testprefix = "/natural_test_embed_and_neighbor_acc_"
testpostfix = "_random_rotate_and_shift_x4.npz"
neighbourkey = ["train_neighbor_avg_acc", "test_neighbor_avg_acc"]
models = ["vgg16", "resnet", "wrn"]
prefix = [trainprefix, testprefix]
postfix = [trainpostfix, testpostfix]

import matplotlib.pyplot as plt



'''
table 2 in paper
'''
def get_correlation():
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
table 3 in paper
'''
def get_correlation2():
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
figure 7 in paper
'''
def get_neighbour_accuracy_change_hist():
    fig, axs = plt.subplots(2, 3, figsize=(30, 20), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0})
    for di in range(len(dataset)):
        for mode in [0, 1]:
            d = dataset[di]
            # m1,m2 can be changed to any two of the three models: models[0],model[1],models[2]
            # currently it is vgg16 vs resnet.
            # "models" is a global variable defined at the top of this script.
            m1 = models[0]
            m2 = models[1]

            ind = mode * 3 + di
            ax = axs[mode][di]
            raw_data = np.load(data_path + d + prefix[mode] + m1 + postfix[mode])
            raw_data2 = np.load(data_path + d + prefix[mode] + m2 + postfix[mode])


            data_to_plot = []
            data_to_plot += list(raw_data[neighbourkey[mode]] - raw_data2[neighbourkey[mode]])
            count = 0
            for dp in data_to_plot:
                if dp >= -0.2 and dp <= 0.2:
                    count += 1
            print("dataset: "+d)
            if mode == 0:
                print("per-traindata neighbour accuracy change between -0.2 to 0.2")
            else:
                print("per-testdata neighbour accuracy change between -0.2 to 0.2")
            print(m1 + " vs " + m2)
            print("percentage of data: " + str(count/len(raw_data[neighbourkey[mode]])))
            (n, bins, patches) = ax.hist(data_to_plot, color='grey', edgecolor='black',
                     bins=int(20))

            ax.tick_params(axis='x', labelsize=45)
            ax.tick_params(axis='y', labelsize=45)
            ax.set_xticks([-0.8, -0.4, 0, 0.4, 0.8])

            if mode == 1:
                if di == 1:
                    ax.set_xlabel("Neighbour Accuracy Difference\n" +  dataset[di], fontsize=60)
                else:
                    ax.set_xlabel("\n" + dataset[di], fontsize=60)

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
    print("figure 7 in paper is saved in 'diff.pdf'")


'''
figure 6 in paper
'''
def get_histogram(dataset="cifar10"):
    fig, axs = plt.subplots(2, 3, figsize=(30, 20), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0.05})
    xlabeltext = ['Train Data Neighbour Accuracy', 'Test Data Neighbour Accuracy']

    d = dataset
    for mi in range(len(models)):
        for mode in [0, 1]:
            ax = axs[mode][mi]
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
                    ax.set_xlabel("Neighbour Accuracy\n" + models[mi], fontsize=60)
                else:
                    ax.set_xlabel("\n" + models[mi], fontsize=60)
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
    plt.savefig(d + ".pdf", bbox_inches='tight', dpi = 1000)
    print("figure 6 in paper is saved in '<dataset_name>.pdf'")

'''
figure 6 in paper
'''
def get_histogram_without_100(dataset="cifar10"):
    fig, axs = plt.subplots(2, 3, figsize=(30, 20), sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0.05})
    xlabeltext = ['Train Data Neighbour Accuracy', 'Test Data Neighbour Accuracy']
    d = dataset
    for mi in range(len(models)):
        for mode in [0, 1]:
            ax = axs[mode][mi]
            m = models[mi]
            ind = mode*3 + mi
            raw_data = np.load(data_path + d + prefix[mode] + m + postfix[mode])
            neighbour_accuracy = raw_data[neighbourkey[mode]]
            data_to_plot = []
            for i in range(len(neighbour_accuracy)):
                if neighbour_accuracy[i]!= 1.0:
                    data_to_plot.append(neighbour_accuracy[i])

            (n, bins, patches) = ax.hist(data_to_plot, color='grey', edgecolor='black',
                                          bins=int(19), range=(min(data_to_plot),0.95))
            ax.tick_params(axis='x', labelsize=45)
            ax.tick_params(axis='y', labelsize=45)
            ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])

            if mode == 1:
                if mi == 1:
                    ax.set_xlabel("Neighbour Accuracy\n" + models[mi], fontsize=60)
                else:
                    ax.set_xlabel("\n" + models[mi], fontsize=60)
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
    plt.savefig(d + "_withoutlastbin.pdf", bbox_inches='tight', dpi = 1000)
    print("figure 6 in paper is saved in '<dataset_name>_withoutlastbin.pdf'")

if __name__ == '__main__':
    print("--------------------------------------------------")
    get_histogram("cifar10") #figure 6 in rq1. the arguments can be "fmnist", "cifar10" or "svhn".
    print("--------------------------------------------------")
    get_histogram_without_100("cifar10") #figure 6 in rq1. the arguments can be "fmnist", "cifar10" or "svhn".
    print("--------------------------------------------------")
    get_neighbour_accuracy_change_hist() #figure 7 in rq1
    print("--------------------------------------------------")
    get_correlation() #table 2 in rq1
    print("--------------------------------------------------")
    get_correlation2() #table 3 in rq1
