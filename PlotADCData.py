import matplotlib.font_manager as fm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time
from scipy import signal
CURRENT_TIME = time.strftime('%Y%m%d_%H%M%S')

font_times = fm.FontProperties(family='Times New Roman', stretch=0)
current_path = os.path.dirname(__file__)
sys.path.append(os.path.join(current_path, '..'))

TITLE_FONT = 25
LABEL_FONT = 22
LEGEND_FONT = 20
TICK_FONT = 20


def save_yaml(save_path, current_time, cfg):
    with open(os.path.join(SAVE_PATH, CURRENT_TIME + '_' + 'config.yaml'), 'w', encoding='utf-8') as f:
        for k, v in cfg.items():
            f.write(k + ': ' + str(v) + '\n')


def hanning(num_order, ctrl='symmetric'):
    if ctrl == 'periodic':
        num_point = num_order + 1
    elif ctrl == 'symmetric':
        num_point = num_order
    else:
        num_point = num_order
        print('window ctrl is wrong.\n')

    n = np.arange(num_point)
    win_all = 0.5 * (1 - np.cos(2 * np.pi * n / (num_point - 1)))
    win = win_all[:num_order]

    return win


def chebyshev(num_order, param):
    win = signal.chebwin(num_order, param)
    return win


def mkdir(file_name):
    save_path = os.path.join(current_path, file_name)
    folder = os.path.exists(save_path)
    if not folder:
        os.makedirs(save_path)
        print('Folder has been created: ' + save_path)
    else:
        print('Folder has existed.')
    return save_path


def create_multi_fig(fig_size=(55, 30), fig_dpi=30):
    fig = plt.figure(figsize=fig_size, dpi=fig_dpi)
    plt.subplots_adjust(left=0.03, bottom=0.05, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
    return fig


def plot_sub_fig(x_axis, y, fig, num_row, num_col, sub_fig,
                 title, x_label, y_label, x_lim=[0], y_lim=[0],
                 title_font=TITLE_FONT, label_font=LABEL_FONT, tick_font=TICK_FONT, legend_font=LEGEND_FONT):
    ax = fig.add_subplot(num_row, num_col, sub_fig)
    ax.set_title('Fig.' + str(sub_fig) + ' - ' + title,
                 fontsize=title_font, fontproperties=font_times)
    ax.set_xlabel(x_label, fontsize=label_font, fontproperties=font_times)
    ax.set_ylabel(y_label, fontsize=label_font, fontproperties=font_times)
    assert np.ndim(y) <= 2
    if y.ndim == 1:
        ax.plot(x_axis, y, 'b-', linewidth=1)
    else:
        for k in range(np.shape(y)[0]):
            ax.plot(x_axis, y[k], 'b-', linewidth=0.5)
    # plt.legend(fontsize=legend_font)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    if len(x_lim) == 2:
        plt.xlim(x_lim[0], x_lim[1])
    else:
        pass

    if len(y_lim) == 2:
        plt.ylim(y_lim[0], y_lim[1])
    else:
        pass
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


def plot_surface_sub_fig(x_axis, y_axis, data, fig, num_row, num_col, sub_fig,
                         title, x_label, y_label, x_lim=[0], y_lim=[0], view_init=[35, 25],
                         title_font=TITLE_FONT, label_font=LABEL_FONT, tick_font=TICK_FONT, legend_font=LEGEND_FONT):
    assert len(view_init) == 2
    assert np.ndim(data) == 2
    ax = fig.add_subplot(num_row, num_col, sub_fig, projection="3d")
    ax.view_init(view_init[0], view_init[1])
    ax.plot_surface(x_axis, y_axis, data, rstride=8, cstride=8, alpha=0.3)
    ax.set_title('Fig.' + str(sub_fig) + ' - ' + title,
                 fontsize=title_font, fontproperties=font_times)
    ax.set_xlabel(x_label, fontsize=label_font, fontproperties=font_times)
    ax.set_ylabel(y_label, fontsize=label_font, fontproperties=font_times)
    plt.tick_params(labelsize=tick_font)
    plt.grid('on')
    if len(x_lim) == 2:
        plt.xlim(x_lim[0], x_lim[1])
    else:
        pass

    if len(y_lim) == 2:
        plt.ylim(y_lim[0], y_lim[1])
    else:
        pass
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]


def load_raw_adc_data(data_path, file_name, cfg):
    num_frame = cfg['num_frame']
    num_chirp = cfg['num_chirp']
    num_antenna = cfg['num_antenna']
    num_sample = cfg['num_sample']
    v_array = cfg['v_array']
    raw_data = np.fromfile(os.path.join(data_path, file_name), dtype=np.int16).reshape(num_frame,
                                                                                       num_chirp,
                                                                                       num_antenna,
                                                                                       num_sample)
    print('Raw ADC data shape: ', raw_data.shape)

    assert isinstance(v_array, int)
    if v_array > 1:
        data = np.zeros((num_frame, int(num_chirp / v_array), num_antenna * v_array, num_sample),
                        dtype=np.int16)
        for id in range(v_array):
            data[:, :, (id * num_antenna):(id * num_antenna + num_antenna), :] = raw_data[:, id::v_array, :, :].copy()
        print('ADC data of virtual array shape: ', np.shape(data))
    elif v_array == 1:
        data = raw_data.copy()
    else:
        print('virtual array input is wrong.')
        data = raw_data.copy()

    return raw_data, data


def dc_cal(data):
    num_chirp, num_rx, num_sample = np.shape(data)
    dc_data = np.zeros((num_chirp, num_rx))
    data_cal = np.zeros(np.shape(data))
    for c in range(num_chirp):
        for m in range(num_rx):
            dc_data[c, m] = np.average(data[c, m, :])
            data_cal[c, m, :] = data[c, m, :] - dc_data[c, m]
    return data_cal, dc_data


def dc_cal_multi_frame(data):
    if np.shape(data) == 3:
        num_chirp, num_rx, num_sample = np.shape(data)
        data_cal, dc_data = dc_cal(data)
        data_cal.reshape(1, num_chirp, num_rx, num_sample)
        dc_data.reshape(1, num_chirp, num_rx)
    else:
        num_frame, num_chirp, num_rx, num_sample = np.shape(data)
        dc_data = np.zeros((num_frame, num_chirp, num_rx))
        data_cal = np.zeros(np.shape(data))
        for f in range(num_frame):
            data_cal[f, :, :, :], dc_data[f, :, :] = dc_cal(data[f])

    return data_cal, dc_data


if __name__ == '__main__':

    CFG = {
        # data info
        'data_path': 'C:/Calterah/CalterahClient/data/20221215-142609',
        'file_name': '_sampling0.dat',

        # data configuration
        'num_frame': 10,
        'num_chirp': 256,
        'num_antenna': 4,
        'num_sample': 1024,
        'v_array': 1,

        # fft ctrl
        'sample_rate': 50,
        'r_fft_factor': 1,
        'v_fft_factor': 1,

        # plot ctrl
        'show_frame': 5,
        'average_frame': True,
        'dc_cal': False,
        'pdf_enable': False,
    }

    NUM_FRAME = CFG['num_frame']
    DATA_PATH = CFG['data_path']
    FILE_NAME = CFG['file_name']

    if CFG['pdf_enable']:
        SAVE_PATH = mkdir(FILE_NAME)

    SHOW_FRAME = CFG['show_frame']
    SAMPLE_RATE = CFG['sample_rate'] * 1e6

    """##############################################################################################################"""
    """ Time Domain """
    TS = 1 / SAMPLE_RATE
    T_AXIS = np.arange(CFG['num_sample']) * TS * 1e6
    T_AXIS_N = np.arange(CFG['num_sample'])

    RAW_DATA, DATA = load_raw_adc_data(DATA_PATH, FILE_NAME, CFG)

    if CFG['dc_cal']:
        DATA_DC_CAL, DC_DATA = dc_cal_multi_frame(DATA)
        print('[DC info]: ', DC_DATA[SHOW_FRAME, :, :])
        print(DC_DATA[SHOW_FRAME, :, :])

    """ Plot """
    FIG_TIME = create_multi_fig(fig_size=(80, 30))
    NUM_VIRTUAL_ANTENNA = np.shape(DATA)[2]
    NUM_ROW = 4
    NUM_COL = NUM_VIRTUAL_ANTENNA
    for k in range(NUM_VIRTUAL_ANTENNA):
        SUB_FIG = k + 1
        RX = k
        TITLE = 'Chirp 0' + ' - ' + 'Rx' + str(RX)
        X_LABEL = 'Time - us'
        Y_LABEL = 'Amplitude'
        if CFG['dc_cal']:
            plot_sub_fig(T_AXIS_N, DATA_DC_CAL[SHOW_FRAME, 0, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)
        else:
            plot_sub_fig(T_AXIS_N, DATA[SHOW_FRAME, 0, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)

        SUB_FIG = NUM_VIRTUAL_ANTENNA + k + 1

        RX = k
        TITLE = 'Chirp 0 - 2' + ' - ' + 'Rx' + str(RX)
        X_LABEL = 'Time - us'
        Y_LABEL = 'Amplitude'
        if CFG['dc_cal']:
            plot_sub_fig(T_AXIS_N, DATA_DC_CAL[SHOW_FRAME, 0:3, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)
        else:
            plot_sub_fig(T_AXIS_N, DATA[SHOW_FRAME, 0:3, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)

        SUB_FIG = 2 * NUM_VIRTUAL_ANTENNA + k + 1
        RX = k
        TITLE = 'Chirp 0 - 9' + ' - ' + 'Rx' + str(RX)
        X_LABEL = 'Time - us'
        Y_LABEL = 'Amplitude'
        if CFG['dc_cal']:
            plot_sub_fig(T_AXIS_N, DATA_DC_CAL[SHOW_FRAME, 0:10, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)
        else:
            plot_sub_fig(T_AXIS_N, DATA[SHOW_FRAME, 0:10, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)

        SUB_FIG = 3 * NUM_VIRTUAL_ANTENNA + k + 1
        RX = k
        TITLE = 'All Chirp' + ' - ' + 'Rx' + str(RX)
        X_LABEL = 'Time - us'
        Y_LABEL = 'Amplitude'
        if CFG['dc_cal']:
            plot_sub_fig(T_AXIS_N, DATA_DC_CAL[SHOW_FRAME, :, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)
        else:
            plot_sub_fig(T_AXIS_N, DATA[SHOW_FRAME, :, RX, :], FIG_TIME, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)
    if CFG['pdf_enable']:
        FIG_TIME.savefig(os.path.join(SAVE_PATH, CURRENT_TIME + '_' + 'Time_Frame' + str(SHOW_FRAME) + '.pdf'))

    """##############################################################################################################"""
    """ window """
    NUM_CHIRP_PER_VARRAY = np.shape(DATA)[1]
    # WIN_SAMPLE = hanning(CFG['num_sample'], 'periodic')
    WIN_SAMPLE = chebyshev(CFG['num_sample'], 80)
    WIN_SAMPLE = WIN_SAMPLE / np.sqrt(np.sum(WIN_SAMPLE * WIN_SAMPLE) / len(WIN_SAMPLE))
    # WIN_CHIRP = hanning(NUM_CHIRP_PER_VARRAY, 'periodic')
    WIN_CHIRP = chebyshev(NUM_CHIRP_PER_VARRAY, 80)
    WIN_CHIRP = WIN_CHIRP / np.sqrt(np.sum(WIN_CHIRP * WIN_CHIRP) / len(WIN_CHIRP))
    WIN_ANTENNA = np.ones(NUM_VIRTUAL_ANTENNA)
    WIN = np.outer(WIN_CHIRP, np.outer(WIN_ANTENNA, WIN_SAMPLE)).reshape(NUM_CHIRP_PER_VARRAY,
                                                                         NUM_VIRTUAL_ANTENNA,
                                                                         CFG['num_sample'])

    """##############################################################################################################"""
    """ 1D-FFT """
    RANGE_FFT_SIZE = CFG['num_sample'] * CFG['r_fft_factor']
    RANGE_F_AXIS = np.arange(RANGE_FFT_SIZE // 2) / RANGE_FFT_SIZE * CFG['sample_rate']
    RANGE_F_AXIS_N = np.arange(RANGE_FFT_SIZE // 2)

    if CFG['dc_cal']:
        DATA_R_F = np.fft.fft(DATA_DC_CAL[SHOW_FRAME] * WIN, axis=-1, n=RANGE_FFT_SIZE)
        DATA_R_F_dB = 20 * np.log10(np.abs(DATA_R_F))
    else:
        DATA_R_F = np.fft.fft(DATA[SHOW_FRAME] * WIN, axis=-1, n=RANGE_FFT_SIZE)
        DATA_R_F_dB = 20 * np.log10(np.abs(DATA_R_F))

    """ Plot """
    FIG_RFFT = create_multi_fig(fig_size=(80, 10))
    NUM_ROW = 1
    NUM_COL = NUM_VIRTUAL_ANTENNA
    for k in range(NUM_VIRTUAL_ANTENNA):
        SUB_FIG = k + 1
        RX = k
        TITLE = 'All Chirp' + ' - ' + 'Rx' + str(RX)
        X_LABEL = 'Time - us'
        Y_LABEL = 'Amplitude'
        plot_sub_fig(RANGE_F_AXIS, DATA_R_F_dB[:, RX, :RANGE_FFT_SIZE//2], FIG_RFFT, NUM_ROW, NUM_COL, SUB_FIG,
                     TITLE, X_LABEL, Y_LABEL)

    if CFG['pdf_enable']:
        FIG_RFFT.savefig(os.path.join(SAVE_PATH, CURRENT_TIME + '_' + 'RFFT_Frame' + str(SHOW_FRAME) + '.pdf'))

    """##############################################################################################################"""
    """ 2D-FFT """
    VEL_FFT_SIZE = NUM_CHIRP_PER_VARRAY * CFG['v_fft_factor']
    VEL_F_AXIS = np.arange(-VEL_FFT_SIZE // 2, VEL_FFT_SIZE // 2) / VEL_FFT_SIZE
    VEL_F_AXIS_N = np.arange(-VEL_FFT_SIZE // 2, VEL_FFT_SIZE // 2)

    DATA_R_V_F = np.fft.fftshift(np.fft.fft(DATA_R_F, axis=-3, n=VEL_FFT_SIZE), axes=-3)
    DATA_R_V_F_dB = 20 * np.log10(np.abs(DATA_R_V_F))

    """ Plot """
    FIG_RVFFT = create_multi_fig(fig_size=(80, 20))

    NUM_ROW = 2
    NUM_COL = NUM_VIRTUAL_ANTENNA
    for k in range(NUM_VIRTUAL_ANTENNA):
        SUB_FIG = k + 1
        RX = k
        TITLE = '2dFFT-R - ' + 'Rx' + str(RX)
        X_LABEL = 'Range - Bin'
        Y_LABEL = 'Magnitude - dB'
        plot_sub_fig(RANGE_F_AXIS_N, DATA_R_V_F_dB[:, RX, :RANGE_FFT_SIZE//2], FIG_RVFFT, NUM_ROW, NUM_COL, SUB_FIG,
                     TITLE, X_LABEL, Y_LABEL)

        SUB_FIG = NUM_VIRTUAL_ANTENNA + k + 1
        RX = k
        TITLE = '2dFFT-V - ' + 'Rx' + str(RX)
        X_LABEL = 'Velocity - Bin'
        Y_LABEL = 'Magnitude - dB'
        plot_sub_fig(VEL_F_AXIS_N, DATA_R_V_F_dB[:, RX, :RANGE_FFT_SIZE//2].T, FIG_RVFFT, NUM_ROW, NUM_COL, SUB_FIG,
                     TITLE, X_LABEL, Y_LABEL)

    if CFG['pdf_enable']:
        FIG_RVFFT.savefig(os.path.join(SAVE_PATH, CURRENT_TIME + '_' + 'RVFFT_Frame' + str(SHOW_FRAME) + '.pdf'))

    """ Plot """
    FIG_RVFFT2d = create_multi_fig(fig_size=(80, 15))
    NUM_ROW = 1
    NUM_COL = NUM_VIRTUAL_ANTENNA
    X, Y = np.meshgrid(RANGE_F_AXIS_N, VEL_F_AXIS_N)
    for k in range(NUM_VIRTUAL_ANTENNA):
        SUB_FIG = k + 1
        RX = k
        TITLE = '2dFFT-R - ' + 'Rx' + str(RX)
        X_LABEL = 'Range - Bin'
        Y_LABEL = 'Velocity - Bin'
        plot_surface_sub_fig(X, Y, DATA_R_V_F_dB[:, RX, :RANGE_FFT_SIZE//2],
                             FIG_RVFFT2d, NUM_ROW, NUM_COL, SUB_FIG,
                             TITLE, X_LABEL, Y_LABEL)
    FIG_RVFFT2d.tight_layout()

    if CFG['pdf_enable']:
        FIG_RVFFT2d.savefig(os.path.join(SAVE_PATH, CURRENT_TIME + '_' + 'RVFFT_Surface_Frame'
                                         + str(SHOW_FRAME) + '.pdf'))
    if CFG['pdf_enable']:
        save_yaml(SAVE_PATH, CURRENT_TIME, CFG)

    """##############################################################################################################"""
    """ Average power """
    if CFG['average_frame']:
        if CFG['dc_cal']:
            DATA_ALL_R_F = np.fft.fft(DATA_DC_CAL * WIN, axis=-1, n=RANGE_FFT_SIZE)
            DATA_ALL_R_F_SUM = 1 / NUM_FRAME * np.sum(np.abs(DATA_ALL_R_F) ** 2, axis=0)
            DATA_ALL_R_F_SUM_dB = 10 * np.log10(np.abs(DATA_ALL_R_F_SUM))

            DATA_ALL_R_V_F = np.fft.fftshift(np.fft.fft(DATA_ALL_R_F, axis=-3, n=VEL_FFT_SIZE), axes=-3)
            DATA_ALL_R_V_F_SUM = 1 / NUM_FRAME * np.sum(np.abs(DATA_ALL_R_V_F) ** 2, axis=0)
            DATA_ALL_R_V_F_SUM_dB = 10 * np.log10(np.abs(DATA_ALL_R_V_F_SUM))

        else:
            DATA_ALL_R_F = np.fft.fft(DATA * WIN, axis=-1, n=RANGE_FFT_SIZE)
            DATA_ALL_R_F_SUM = 1 / NUM_FRAME * np.sum(np.abs(DATA_ALL_R_F) ** 2, axis=0)
            DATA_ALL_R_F_SUM_dB = 10 * np.log10(np.abs(DATA_ALL_R_F_SUM))

            DATA_ALL_R_V_F = np.fft.fftshift(np.fft.fft(DATA_ALL_R_F, axis=-3, n=VEL_FFT_SIZE), axes=-3)
            DATA_ALL_R_V_F_SUM = 1 / NUM_FRAME * np.sum(np.abs(DATA_ALL_R_V_F) ** 2, axis=0)
            DATA_ALL_R_V_F_SUM_dB = 10 * np.log10(np.abs(DATA_ALL_R_V_F_SUM))

        FIG_RVAFFT = create_multi_fig(fig_size=(80, 35))

        NUM_ROW = 3
        NUM_COL = NUM_VIRTUAL_ANTENNA
        for k in range(NUM_VIRTUAL_ANTENNA):
            SUB_FIG = k + 1
            RX = k
            TITLE = 'Average 1dFFT' + ' - ' + 'Rx' + str(RX)
            X_LABEL = 'Time - us'
            Y_LABEL = 'Amplitude'
            plot_sub_fig(RANGE_F_AXIS_N, DATA_ALL_R_F_SUM_dB[:, RX, :RANGE_FFT_SIZE // 2],
                         FIG_RVAFFT, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)

            SUB_FIG = NUM_VIRTUAL_ANTENNA + k + 1
            RX = k
            TITLE = 'Average 2dFFT-R - ' + 'Rx' + str(RX)
            X_LABEL = 'Range - Bin'
            Y_LABEL = 'Magnitude - dB'
            plot_sub_fig(RANGE_F_AXIS_N, DATA_ALL_R_V_F_SUM_dB[:, RX, :RANGE_FFT_SIZE // 2],
                         FIG_RVAFFT, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)

            SUB_FIG = 2 * NUM_VIRTUAL_ANTENNA + k + 1
            RX = k
            TITLE = 'Average 2dFFT-V - ' + 'Rx' + str(RX)
            X_LABEL = 'Velocity - Bin'
            Y_LABEL = 'Magnitude - dB'
            plot_sub_fig(VEL_F_AXIS_N, DATA_ALL_R_V_F_SUM_dB[:, RX, :RANGE_FFT_SIZE // 2].T,
                         FIG_RVAFFT, NUM_ROW, NUM_COL, SUB_FIG,
                         TITLE, X_LABEL, Y_LABEL)

        if CFG['pdf_enable']:
            FIG_RVAFFT.savefig(os.path.join(SAVE_PATH, CURRENT_TIME + '_' + 'RVAFFT_Frame' + str(SHOW_FRAME) + '.pdf'))

    """##############################################################################################################"""
    if not CFG['pdf_enable']:
        plt.show()
    # max_index = np.unravel_index(np.argmax(sig_mat_dB, axis=None), sig_mat_dB.shape)
