# from unittest import TestCase
# import numpy as np
# from dat_analysis.dat_object.attributes import square_entropy as SE
# from dat_analysis.hdf_util import NotFoundInHdfError
# from tests import helpers
#
# output_dir = 'Outputs/SquareEntropy/'
#
#
# class TestSquareEntropy(TestCase):
#     helpers.clear_outputs(output_dir)
#     dat = helpers.init_testing_dat(9111, output_directory=output_dir)
#     S = SE.SquareEntropy(dat)
#
#     AVG_ES = np.array([np.nan, np.nan, 6.86485202e-04, 3.90781398e-04,
#                        - 3.05070723e-04, 1.69951469e-04, - 1.27298579e-04, - 1.75270244e-04,
#                        - 9.95574781e-05, 4.49816519e-04, - 1.89807369e-04, 3.59287890e-04,
#                        2.45136853e-05, - 6.68287984e-05, - 1.39056316e-05, 3.24940661e-05,
#                        7.24466341e-04, 4.84392143e-04, 6.34892839e-04, 7.23638297e-04,
#                        1.15104101e-04, 1.76117787e-04, 8.63863132e-04, 1.19429097e-03,
#                        1.17017417e-03, 9.78796708e-04, 1.03261494e-03, 6.23626075e-04,
#                        4.70399622e-04, 6.82039962e-04, 6.71992953e-04, 6.26006732e-04,
#                        4.92452883e-04, 1.33210532e-03, 1.02224105e-03, 1.67353774e-03,
#                        1.23888215e-03, 2.29178643e-03, 2.30967805e-03, 1.47398501e-03,
#                        1.99916030e-03, 1.86495129e-03, 1.54732985e-03, 2.26818038e-03,
#                        2.06692008e-03, 2.81093355e-03, 3.57242782e-03, 3.66778442e-03,
#                        3.58049758e-03, 4.36596587e-03, 5.27325241e-03, 5.83899735e-03,
#                        6.85194764e-03, 7.82279154e-03, 9.33962395e-03, 1.01805559e-02,
#                        1.22672839e-02, 1.45817277e-02, 1.73588761e-02, 1.94998608e-02,
#                        2.27191800e-02, 2.61460627e-02, 3.08683716e-02, 3.49287950e-02,
#                        3.73538133e-02, 4.09266937e-02, 4.43796718e-02, 4.48241093e-02,
#                        5.18911494e-02, 5.08101682e-02, 4.61014671e-02, 4.13260502e-02,
#                        4.03757761e-02, 1.90874121e-02, 1.39892528e-02, 2.32774900e-03,
#                        - 5.82549322e-03, - 8.66357745e-03, - 1.61046378e-02, - 2.04262449e-02,
#                        - 1.93591577e-02, - 1.99264698e-02, - 1.96786788e-02, - 1.61033829e-02,
#                        - 1.45559275e-02, - 1.36504940e-02, - 1.25371691e-02, - 1.03538541e-02,
#                        - 8.28208653e-03, - 6.83091042e-03, - 6.10784585e-03, - 5.46407979e-03,
#                        - 4.44905107e-03, - 4.40821242e-03, - 4.02013585e-03, - 2.76302961e-03,
#                        - 2.75002460e-03, - 2.61325473e-03, - 2.23081347e-03, - 2.19827931e-03,
#                        - 2.31072194e-03, - 1.83569922e-03, - 1.49798402e-03, - 1.20027922e-03,
#                        - 9.38620743e-04, - 5.74995414e-04, - 7.98033934e-04, - 4.30411425e-04,
#                        - 9.24303293e-04, - 1.11520872e-03, - 6.37167035e-04, - 9.39164349e-04,
#                        - 4.30357527e-04, - 4.97752630e-04, - 5.47639534e-04, - 2.59080642e-04,
#                        - 4.81575519e-04, - 1.10042722e-03, - 3.21475934e-04, - 4.09356933e-05,
#                        2.66025874e-06, - 2.53274931e-04, 3.52965493e-04, - 6.14843784e-04,
#                        1.96178844e-04, - 4.09736569e-04, - 4.59907395e-04, 2.70872632e-04,
#                        3.64776584e-04, - 2.44708709e-04, - 7.52142343e-06, - 1.26001192e-05,
#                        - 6.38788780e-04, 2.30743570e-04, 5.06588341e-04, - 4.66660318e-04,
#                        - 1.76292477e-04, - 1.28766508e-04, 3.25317900e-05, 6.88568576e-05,
#                        - 5.55032203e-04, 3.87720424e-04, - 5.05907940e-05, - 4.15717112e-04,
#                        - 3.02048430e-04, np.nan, np.nan, np.nan])
#
#     def tearDown(self) -> None:
#         """Runs AFTER every test
#         Check that HDF is left closed
#         """
#         pass
#
#     def test_square_awg(self):
#         awg = self.S.square_awg
#         self.assertEqual(4, len(awg.AWs[0][0]))
#
#     def test_default_input(self):
#         inp = self.S.default_Input
#         print(inp.full_wave_masks.shape, inp.x_array.shape, inp.avg_nans)
#         self.assertEqual(((4, 72816), (72816,), False), (inp.full_wave_masks.shape, inp.x_array.shape, inp.avg_nans))
#
#     def test_default_process_params(self):
#         pp = self.S.default_ProcessParams
#         expected = SE.ProcessParams(setpoint_start=None, setpoint_fin=None, cycle_start=None, cycle_fin=None, transition_fit_func=None, transition_fit_params=None)
#         self.assertEqual(expected, pp)
#
#     def test_default_output(self):
#         out = self.S.default_Output
#         print(self.AVG_ES, out.average_entropy_signal)
#         self.assertTrue(np.allclose(self.AVG_ES, out.average_entropy_signal, equal_nan=True, atol=1e-5, rtol=0.01))
#
#     def test_x(self):
#         x = self.S.x
#         expected = np.linspace(-297.575, 302.424, 148)
#         print(x[0], x[-1])
#         self.assertTrue(np.allclose(expected, x, atol=0.01))
#
#     def test_entropy_signal(self):
#         es = self.S.entropy_signal
#         self.assertEqual((50, 148), es.shape)
#
#     def test_avg_entropy_signal(self):
#         avg_es = self.S.avg_entropy_signal
#         expected = self.AVG_ES
#         print(expected, avg_es)
#         self.assertTrue(np.allclose(expected, avg_es, equal_nan=True, atol=1e-5, rtol=0.01))
#
#     def test_get_inputs_non_existing(self):
#         with self.assertRaises(NotFoundInHdfError):
#             inp = self.S.get_Inputs('non_existing')
#
#     def test_get_inputs_existing(self):
#         inp = self.S.get_Inputs(i_sense=self.dat.Data.i_sense / 10, avg_nans=True, save_name='saved_inp')
#         saved_inp = self.S.get_Inputs(name='saved_inp')
#         print(inp.i_sense[0, 1000])
#         self.assertTrue(np.isclose(0.4871748, inp.i_sense[0, 1000]))
#
#     def test_get_inputs_overwrite_existing(self):
#         inp = self.S.get_Inputs(i_sense=self.dat.Data.i_sense / 10, avg_nans=True, save_name='saved_inp')
#         new_inp = self.S.get_Inputs(name='saved_inp', i_sense=self.dat.Data.i_sense, avg_nans=False,
#                                     save_name='saved_inp')
#         loaded_inp = self.S.get_Inputs(name='saved_inp')
#         print(inp.i_sense[0, 0], new_inp.i_sense[0, 0], loaded_inp.i_sense[0, 0])
#         self.assertEqual(new_inp.i_sense[0, 0], loaded_inp.i_sense[0, 0])
#         self.assertNotEqual(inp.i_sense[0, 0], loaded_inp.i_sense[0, 0])
#
#     def test_get_inputs(self):
#         inp = self.S.get_Inputs(i_sense=self.dat.Data.i_sense / 10, avg_nans=True)
#         print(inp.avg_nans, inp.i_sense[0, 0])
#         self.assertTrue(inp.avg_nans)
#
#     def test_get_process_params(self):
#         pp = self.S.get_ProcessParams(setpoint_start=10, setpoint_fin=-2, cycle_start=2)
#         self.assertEqual((10, -2, 2, None), (pp.setpoint_start, pp.setpoint_fin, pp.cycle_start, pp.cycle_fin))
#
#     def test_get_process_params_existing(self):
#         pp = self.S.get_ProcessParams(setpoint_start=10, setpoint_fin=-2, cycle_start=2, save_name='saved_pp')
#         loaded_pp = self.S.get_ProcessParams(name='saved_pp')
#         self.assertEqual((10, -2, 2, None), (loaded_pp.setpoint_start, loaded_pp.setpoint_fin,
#                                              loaded_pp.cycle_start, loaded_pp.cycle_fin))
#
#     def test_get_process_params_changed_existing(self):
#         pp = self.S.get_ProcessParams(setpoint_start=10, setpoint_fin=-2, cycle_start=2, save_name='saved_pp')
#         overwrite_pp = self.S.get_ProcessParams(name='saved_pp', setpoint_start=5, save_name='changed_pp')
#         loaded_pp = self.S.get_ProcessParams(name='changed_pp')
#         self.assertEqual((5, -2, 2, None), (loaded_pp.setpoint_start, loaded_pp.setpoint_fin,
#                                             loaded_pp.cycle_start, loaded_pp.cycle_fin))
#
#     def test_get_outputs(self):
#         out = self.S.get_Outputs()
#         print(self.AVG_ES, out.average_entropy_signal)
#         self.assertTrue(np.allclose(self.AVG_ES, out.average_entropy_signal, equal_nan=True, atol=1e-5, rtol=0.01))
#
#     def test_get_outputs_load(self):
#         pp = self.S.get_ProcessParams(setpoint_start=10, setpoint_fin=-2)
#         out = self.S.get_Outputs(name='new_save', process_params=pp)
#         loaded_out = self.S.get_Outputs(name='new_save')
#         default_out = self.S.get_Outputs()
#         self.assertNotEqual(list(loaded_out.average_entropy_signal), list(default_out.average_entropy_signal))
#
#     def test_get_outputs_overwrite(self):
#         pp = self.S.get_ProcessParams(setpoint_start=10, setpoint_fin=-2)
#         out = self.S.get_Outputs(name='new_save', process_params=pp)
#         loaded_out = self.S.get_Outputs(name='new_save', overwrite=True, check_exists=False)
#         default_out = self.S.get_Outputs()
#         self.assertTrue(
#             np.allclose(loaded_out.average_entropy_signal, default_out.average_entropy_signal, equal_nan=True))
#
#     def test_clear_caches(self):
#         awg = self.S.square_awg
#         self.assertEqual(awg, self.S._square_awg)
#         es = self.S.avg_entropy_signal
#         self.assertTrue(np.allclose(es, self.S._Outputs['default'].average_entropy_signal, equal_nan=True))
#         self.S.clear_caches()
#         self.assertEqual([None, {}], [self.S._square_awg, self.S._Outputs])
#
#     def test_get_SquareEntropy_from_dat(self):
#         se = self.dat.SquareEntropy
#         self.assertEqual(se.dat, self.S.dat)