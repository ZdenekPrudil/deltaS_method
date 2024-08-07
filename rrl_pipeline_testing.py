from fourgp.pipeline.pipeline_rrlyr import *
import pytest

#from unittest import assertTupleEqual

ca_boundary = np.array([3910.00, 3925.00, 3940.00, 3955.00, 3923.67, 3943.67])

# ------------------------------------------------------------------------------

wave = np.array([
       3900.31764262, 3901.21704726, 3902.11451711, 3903.01219343,
       3903.91221942, 3904.81030929, 3905.71074993, 3906.60925355,
       3907.51010902, 3908.40902658, 3909.30815094, 3910.20962877,
       3911.10916736, 3912.01106049, 3912.9110135 , 3913.81332213,
       3914.71368974, 3915.61426448, 3916.51719649, 3917.41818612,
       3918.3215341 , 3919.22293882, 3920.12670297, 3921.02852297,
       3921.93055043, 3922.83493895, 3923.73738198, 3924.64218715,
       3925.54504593, 3926.45026794, 3927.35354267, 3928.25702519,
       3929.16287258, 3930.06677134, 3930.97303606, 3931.87735124,
       3932.78403347, 3933.68876527, 3934.5937052 , 3935.50101382,
       3936.40637066, 3937.31409727, 3938.2198712 , 3939.12801601,
       3940.03420723, 3940.94060692, 3941.84937912, 3942.75619639,
       3943.66538726, 3944.5726223 , 3945.48223204, 3946.38988504,
       3947.29774685, 3948.20798499, 3949.11626505, 3950.02692254,
       3950.93562104, 3951.84669807, 3952.75581521, 3953.66514149,
       3954.57684794, 3955.48659314, 3956.39871962, 3957.30888394,
       3958.22143063, 3959.13201427])

flux = np.array([
        839.82012939,  869.74200439,  898.57171631,  926.06152344,
        942.06347656,  931.1348877 ,  935.86627197,  949.25994873,
        960.63293457, 1013.12988281, 1013.94696045, 1007.37365723,
       1022.50549316, 1027.62011719, 1000.39569092,  980.39593506,
       1005.59033203, 1010.78265381, 1053.92456055, 1046.57421875,
       1040.64733887, 1026.27258301, 1017.40905762, 1026.41418457,
       1032.69506836, 1031.85546875, 1045.83898926, 1047.78759766,
       1057.57543945, 1057.2019043 , 1037.97558594, 1032.83337402,
       1020.87091064, 1002.98956299,  997.32019043,  938.20629883,
        796.75268555,  599.72436523,  481.27435303,  727.72674561,
        974.13336182, 1007.77337646, 1037.77868652, 1040.00952148,
       1060.48132324, 1064.71362305, 1033.3503418 , 1040.2923584 ,
       1039.57897949, 1032.45654297, 1021.00994873, 1025.8548584 ,
       1033.00720215, 1024.87304688, 1024.40930176, 1013.62628174,
        986.03338623,  999.95690918, 1012.55969238,  993.42663574,
        981.46813965,  994.75537109,  984.80859375,  948.02532959,
        944.21496582,  933.16101074])

eflux = np.array([
       1.61719748e+01, 1.65796252e+01, 1.69993454e+01, 1.74319174e+01,
       1.76174032e+01, 1.75855893e+01, 1.75457563e+01, 1.77748012e+01,
       1.80476183e+01, 1.86588295e+01, 1.87726922e+01, 1.87514835e+01,
       1.88527123e+01, 1.89005498e+01, 1.84964099e+01, 1.82773936e+01,
       1.85682515e+01, 1.87027833e+01, 1.91976620e+01, 1.91638864e+01,
       1.89974571e+01, 1.87275433e+01, 1.85145062e+01, 1.85780953e+01,
       1.85821695e+01, 1.85951538e+01, 1.87750411e+01, 1.88027914e+01,
       1.89406515e+01, 1.88303516e+01, 1.86130502e+01, 1.84666997e+01,
       1.82401043e+01, 1.80166730e+01, 2.07933551e+01, 1.98171391e+01,
       1.45207150e+01, 1.77237788e+01, 1.64672044e+01, 9.99999996e+14,
       2.30037228e+01, 1.78869337e+01, 1.82842147e+01, 1.84685547e+01,
       1.86745581e+01, 1.87973798e+01, 1.83772509e+01, 1.84200508e+01,
       1.83980699e+01, 1.83077458e+01, 1.82129188e+01, 1.82725427e+01,
       1.83371417e+01, 1.82340109e+01, 1.81743068e+01, 1.79824890e+01,
       1.76370034e+01, 1.77975802e+01, 1.79461065e+01, 1.77151665e+01,
       1.75849545e+01, 1.77249655e+01, 1.75307707e+01, 1.70574258e+01,
       1.69533989e+01, 1.68190271e+01])

# ------------------------------------------------------------------------------

wave_line = np.array([
       3924.64218715, 3925.54504593, 3926.45026794, 3927.35354267,
       3928.25702519, 3929.16287258, 3930.06677134, 3930.97303606,
       3931.87735124, 3932.78403347, 3933.68876527, 3934.5937052 ,
       3935.50101382, 3936.40637066, 3937.31409727, 3938.2198712 ,
       3939.12801601, 3940.03420723, 3940.94060692, 3941.84937912,
       3942.75619639])

flux_line = np.array([
       1047.78759766, 1057.57543945, 1057.2019043 , 1037.97558594,
       1032.83337402, 1020.87091064, 1002.98956299,  997.32019043,
        938.20629883,  796.75268555,  599.72436523,  481.27435303,
        727.72674561,  974.13336182, 1007.77337646, 1037.77868652,
       1040.00952148, 1060.48132324, 1064.71362305, 1033.3503418 ,
       1040.2923584 ])

eflux_line = np.array([
       1.88027914e+01, 1.89406515e+01, 1.88303516e+01, 1.86130502e+01,
       1.84666997e+01, 1.82401043e+01, 1.80166730e+01, 2.07933551e+01,
       1.98171391e+01, 1.45207150e+01, 1.77237788e+01, 1.64672044e+01,
       9.99999996e+14, 2.30037228e+01, 1.78869337e+01, 1.82842147e+01,
       1.84685547e+01, 1.86745581e+01, 1.87973798e+01, 1.83772509e+01,
       1.84200508e+01])

# ------------------------------------------------------------------------------

flux_cont_flip = np.array([
       0.18398335, 0.15476215, 0.12659265, 0.09971591, 0.08399942,
       0.09446785, 0.08970747, 0.07651871, 0.06529109, 0.01403873,
       0.01307121, 0.01929761, 0.0043924 , 0.        , 0.02557938,
       0.04489232, 0.02017625, 0.01494446, 0.        , 0.        ,
       0.        , 0.        , 0.00761573, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.0024843 , 0.01978378, 0.02515211,
       0.08277219, 0.22092504, 0.4134783 , 0.52923753, 0.28804199,
       0.04680613, 0.01371439, 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.00551684, 0.03241645, 0.01857834, 0.00603216, 0.02464014,
       0.03620898, 0.02298686, 0.03258326, 0.06855072, 0.07212842,
       0.08282725])

eflux_cont_flip = np.array([
       1.57136036e-02, 1.61125101e-02, 1.65232813e-02, 1.69466903e-02,
       1.71300044e-02, 1.71020512e-02, 1.70662961e-02, 1.72920983e-02,
       1.75605781e-02, 1.81584648e-02, 1.82724651e-02, 1.82550184e-02,
       1.83567754e-02, 1.84065806e-02, 1.80161543e-02, 1.78059480e-02,
       1.80924708e-02, 1.82267476e-02, 1.87123166e-02, 1.86826695e-02,
       1.85236751e-02, 1.82636965e-02, 1.80591127e-02, 1.81243196e-02,
       1.81314787e-02, 1.81473442e-02, 1.83261207e-02, 1.83564430e-02,
       1.84942840e-02, 1.83898276e-02, 1.81808107e-02, 1.80410364e-02,
       1.78228120e-02, 1.76075961e-02, 2.03248251e-02, 1.93740237e-02,
       1.41985407e-02, 1.73335978e-02, 1.61075314e-02, 9.78331510e+11,
       2.25092461e-02, 1.75055482e-02, 1.78975248e-02, 1.80811741e-02,
       1.82860944e-02, 1.84096222e-02, 1.80013574e-02, 1.80464812e-02,
       1.80281511e-02, 1.79428266e-02, 1.78530662e-02, 1.79146932e-02,
       1.79812213e-02, 1.78832780e-02, 1.78278919e-02, 1.76428758e-02,
       1.73069944e-02, 1.74676834e-02, 1.76165938e-02, 1.73929927e-02,
       1.72682338e-02, 1.74088270e-02, 1.72211749e-02, 1.67591799e-02,
       1.66599531e-02, 1.65308593e-02])

# ------------------------------------------------------------------------------

test_ew_ls = np.array([ [1,1,1,1,1,1],
                        [2,2,2,2,2,2],
                        [3,3,3,3,3,3],
                        [4,4,4,4,4,4] ]).T

test_ew_ls_bad = np.array([ [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                            [2,2,2,2,2,2],
                            [3,3,3,3,3,3],
                            [4,4,4,4,4,4] ]).T

ca_shift = -0.809808

ca_region = np.where((wave > min(ca_boundary)) & (wave < max(ca_boundary)))

# ------------------------------------------------------------------------------


class Test_center_line:
    def test_normal_case(self):
        assert isinstance(center_line(wave_line, flux_line, eflux_line), float)#, "correct type"

    def test_negative_case_turned_positive(self):
        empty_array = np.array([])
        NoneType = type(None)
        assert isinstance(center_line(empty_array, flux_line, eflux_line), NoneType)


class Test_calculate_rr_ew:
    def test_normal_case(self):
        assert isinstance(calculate_rr_ew(wave, flux_cont_flip, eflux_cont_flip, ca_boundary), float)#, "correct type"

    def test_negative_case_turned_positive(self):
        empty_array = np.array([])
        NoneType = type(None)
        assert isinstance(calculate_rr_ew(wave, empty_array, empty_array, ca_boundary), NoneType)


class Test_calculate_snr:
    def test_normal_case(self):
        assert isinstance(calculate_snr(wave, flux_cont_flip, ca_boundary), float)#, "correct type"

    def test_negative_case_turned_positive(self):
        empty_array = np.array([])
        NoneType = type(None)
        assert isinstance(calculate_snr(wave, empty_array, ca_boundary), NoneType)


class Test_normalize_flux:
    def test_normal_case(self):
        expected_output = (flux_cont_flip, eflux_cont_flip)
        output = normalize_flux(wave, flux, eflux, ca_boundary)
        assert np.allclose(output[0], expected_output[0]) & np.allclose(output[1], expected_output[1])

    def test_negative(self):
        expected_output = (np.ones(len(wave)), np.ones(len(wave)))
        output = normalize_flux(wave, flux, eflux, ca_boundary)
        assert ~np.allclose(output[0], expected_output[0]) | ~np.allclose(output[1], expected_output[1])

    def test_negative_case_turned_positive(self):
        expected_output = (np.array([]), np.array([]))
        output = normalize_flux(wave, np.array([]), eflux, ca_boundary)
        assert np.allclose(output[0], expected_output[0]) & np.allclose(output[1], expected_output[1])


class Test_calculation_of_metallicity:
    def test_normal_case(self):
        assert isinstance(calculation_of_metallicity(test_ew_ls)[0], float)

    def test_normal_case2(self):
        assert isinstance(calculation_of_metallicity(test_ew_ls)[1], float)

    def test_negative_case_turned_positive(self):
        assert isinstance(calculation_of_metallicity(test_ew_ls_bad)[0], float)

    def test_negative_case_turned_positive2(self):
        assert isinstance(calculation_of_metallicity(test_ew_ls_bad)[0], float)


class Test_est_individual_lines:
    def test_normal_case(self):
        test_array = np.empty(NUMBER_OF_ITERATIONS).reshape((NUMBER_OF_ITERATIONS, 1))
        output_list = est_individual_lines(wave, flux, eflux,
                                           [ca_boundary], [ca_region], [ca_shift])[0]
        assert np.shape(output_list) == np.shape(test_array)

    def test_normal_case2(self):
        test_array = np.empty(NUMBER_OF_ITERATIONS).reshape((NUMBER_OF_ITERATIONS, 1))
        output_list = est_individual_lines(wave, flux, eflux,
                                           [ca_boundary], [ca_region], [ca_shift])[1]
        assert np.shape(output_list) == np.shape(test_array)

    def test_negative_case_turned_positive(self):
        output_list = est_individual_lines(np.array([]), flux, eflux,
                                           [ca_boundary], [ca_region], [ca_shift])[0]
        assert np.shape(output_list) != np.shape(np.array([]))

    def test_negative_case_turned_positive2(self):
        output_list = est_individual_lines(np.array([]), flux, eflux,
                                           [ca_boundary], [ca_region], [ca_shift])[1]
        assert np.shape(output_list) != np.shape(np.array([]))



class Test_multi_proc_function:
    def test_normal_case(self):
        assert isinstance(multi_proc_function(wave, flux, eflux,
                                              [ca_boundary], [ca_region], [ca_shift])[0][0],
                                              float)

    def test_normal_case2(self):
        assert isinstance(multi_proc_function(wave, flux, eflux,
                                              [ca_boundary], [ca_region], [ca_shift])[1][0],
                                              float)

    def test_negative_case_turned_positive(self):
        assert isinstance(multi_proc_function(wave, np.array([]), eflux,
                                              [ca_boundary], [ca_region], [ca_shift])[0][0],
                                              float)

    def test_negative_case_turned_positive2(self):
        assert isinstance(multi_proc_function(wave, np.array([]), eflux,
                                              [ca_boundary], [ca_region], [ca_shift])[1][0],
                                              float)



"""
# For specific runs

def test_negative_case_turned_positive():
    empty_array = np.array([])
    NoneType = type(None)
    assert isinstance(center_line(empty_array, flux_line, eflux_line), NoneType)
"""
