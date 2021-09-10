"""
Steps:
Read the npz files from the designated directory.
Read the 50 values for each window - do this for each file. In the end, you will have 100 x 50.
Then run the kernel smoothing operator on the 100 x 50 for each window. Then sample 100 values from this PDF.
Finally you will have 100 values for each window.
Save everything into a dictionary and to disk.
"""


"""
Implementing Kernel density estimation of the null_distribution results.
"""
from scipy import stats
import glob
import numpy as np

def do_kernel_smoothing(vals, resample_size=100):
    kernel = stats.gaussian_kde(vals)
    return kernel.resample(resample_size)

#-------------------------------------------------------------------------------------
"""
This section takes all the permutation test results that are stored in a directory and then collates them into an array.
Select the block and then run it in console.
"""


def single_window_perm():
    """
    This function is for the non-tgm analysis.
    NOTE: The age group, input files directory and the output file directory have to be hardcoded.
    """
    numpy_vars = {}
    i = 0
    for np_name in glob.glob('G:\jw_lab\jwlab_eeg\\regression\permutation_test_results\\12m fine tuned res\*.np[yz]'):
        numpy_vars[i] = np.load(np_name, allow_pickle=True)
        i += 1

    res = []
    i = 0
    for key, val in numpy_vars.items():
        print(i)
        try:
            temp = val['arr_0'].tolist()
        except:
            print('File id: ', numpy_vars[i].fid)
        i +=1
        res.append(temp)


    # All window values -> 100 x 50 for each window.
    all_wind_vals = {}
    for window in range(111):
        all_wind_vals[window] = []

    # Now iterate over res and append all values to the respective window.
    for wind in range(111):
        for d in res:
            vals = d[0] # This gives you the dictionary which contains 50 values for all windows.
            # print(vals[wind])
            all_wind_vals[wind].extend([np.mean(vals[wind])])
    #
    # np.savez_compressed('G:\jw_lab\jwlab_eeg\\regression\permuted_npz_processed\kde\9m_fine_res_w2v_kde_null_dist.npz', all_wind_vals)
    # Now get kernel pdfs for each window.

    smoothed_perms = {}
    for wind in range(111):
        smoothed_vals = do_kernel_smoothing(all_wind_vals[wind], 100)
        smoothed_perms[wind] = smoothed_vals.tolist()[0]
    #
    #
    final_res_arr = np.array(smoothed_perms)
    # Now save the 'final_res' object to disk.
    np.savez_compressed('G:\jw_lab\jwlab_eeg\\regression\permuted_npz_processed\kde\\12m_fine_res_w2v_kde_null_dist.npz', final_res_arr)




def tgm_perm():
    """
    This function is for the the permutated TGM analysis.
    """
    # Load the data.
    i = 0
    numpy_tgms = {}
    for np_name in glob.glob('G:\jw_lab\jwlab_eeg\\regression\permutation_test_results\\tgms\Across\\12_to_9m_pre_w2v_xval\*.np[yz]'):
        numpy_tgms[i] = np.load(np_name, allow_pickle=True)
        i += 1

    tgms = []
    i = 0
    for key, val in numpy_tgms.items():
        try:
            temp = val['arr_0']
            tgms.append(temp)
        except:
            print("Corrupted file: ",numpy_tgms[i].fid)
        i += 1

    tgm_avg = {}
    for i in range(len(tgms)):
        permutation = tgms[i]
        tgm_avg[i] = np.mean(permutation, axis=0)

    ###### final_res_arr = np.array(tgm_avg)
    # Now save the 'final_res' object to disk.
    np.savez_compressed('G:\jw_lab\jwlab_eeg\\regression\permuted_npz_processed\\tgms\Across\\non_kde\\12_to_9m_pre_w2v\\12_to_9m_pre_w2v_xval.npz', tgm_avg)
