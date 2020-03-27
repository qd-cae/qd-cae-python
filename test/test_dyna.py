
import filecmp
import os
import sys
import unittest as unittest

import numpy as np
from qd.cae.dyna import *


class TestDynaModule(unittest.TestCase):

    """Tests the dyna module."""

    def is_almost_equal(self, val1, val2, tolerance):

        if(np.abs(val1 - val2) < tolerance):
            return True
        else:
            return False

    def assertCountEqual(self, *args, **kwargs):
        '''Redefine because it does not exist in python2 unittest'''

        if (sys.version_info[0] >= 3):
            super(TestDynaModule, self).assertCountEqual(*args, **kwargs)
        else:
            super(TestDynaModule, self).assertItemsEqual(*args, **kwargs)

    def test_dyna_d3plot(self):

        d3plot_filepath = "test/d3plot"
        d3plot_modes = ["inner", "mid", "outer", "max", "min", "mean"]
        d3plot_vars = ["disp", "vel", "accel",
                       "stress", "strain", "plastic_strain", "stress", "stress_mises",
                       "history 1 shell", "history 1 solid"]
        element_result_list = [None,
                               "plastic_strain",
                               lambda elem: elem.get_plastic_strain()[-1]]
        part_ids = [1]

        # D3plot loading/unloading
        d3plot = D3plot(d3plot_filepath)
        for mode in d3plot_modes:  # load every var with every mode
            d3plot = D3plot(d3plot_filepath)
            vars2 = ["%s %s" % (var, mode) for var in d3plot_vars]
            for var in vars2:
                d3plot.read_states(var)
            d3plot = D3plot(d3plot_filepath, read_states=vars2)  # all at once
        d3plot = D3plot(d3plot_filepath)
        d3plot.read_states("disp")
        d3plot.clear("disp")
        self.assertEqual(len(d3plot.get_nodeByIndex(1).get_disp()), 0)
        d3plot.read_states("vel")
        d3plot.clear("vel")
        self.assertEqual(len(d3plot.get_nodeByIndex(1).get_vel()), 0)
        d3plot.read_states("accel")
        d3plot.clear("accel")
        self.assertEqual(len(d3plot.get_nodeByIndex(1).get_accel()), 0)
        d3plot.read_states("energy")
        d3plot.clear("energy")
        self.assertEqual(len(d3plot.get_elementByID(
            Element.shell, 1).get_energy()), 0)
        d3plot.read_states("plastic_strain")
        d3plot.clear("plastic_strain")
        self.assertEqual(len(d3plot.get_elementByID(
            Element.shell, 1).get_plastic_strain()), 0)
        d3plot.read_states("stress")
        d3plot.clear("stress")
        self.assertEqual(
            len(d3plot.get_elementByID(Element.shell, 1).get_stress()), 0)
        d3plot.read_states("strain")
        d3plot.clear("strain")
        self.assertEqual(
            len(d3plot.get_elementByID(Element.shell, 1).get_strain()), 0)
        d3plot.read_states("history shell 1")
        d3plot.clear("history")
        self.assertEqual(
            len(d3plot.get_elementByID(Element.shell, 1).get_history_variables()), 0)
        d3plot.read_states("stress_mises")
        d3plot.clear("stress_mises")
        self.assertEqual(len(d3plot.get_elementByID(
            Element.shell, 1).get_stress_mises()), 0)
        # default mode (mode=mean)
        d3plot = D3plot(d3plot_filepath, read_states=d3plot_vars)
        d3plot.clear()
        self.assertEqual(len(d3plot.get_nodeByIndex(1).get_disp()), 0)
        self.assertEqual(len(d3plot.get_nodeByIndex(1).get_vel()), 0)
        self.assertEqual(len(d3plot.get_nodeByIndex(1).get_accel()), 0)
        self.assertEqual(
            len(d3plot.get_elementByID(Element.shell, 1).get_energy()), 0)
        self.assertEqual(len(d3plot.get_elementByID(
            Element.shell, 1).get_plastic_strain()), 0)
        self.assertEqual(
            len(d3plot.get_elementByID(Element.shell, 1).get_stress()), 0)
        self.assertEqual(
            len(d3plot.get_elementByID(Element.shell, 1).get_strain()), 0)
        self.assertEqual(
            len(d3plot.get_elementByID(Element.shell, 1).get_history_variables()), 0)
        # D3plot functions
        # default mode (mode=mean)
        d3plot = D3plot(d3plot_filepath, read_states=d3plot_vars)
        self.assertEqual(d3plot.get_nNodes(), 4915)
        self.assertEqual(len(d3plot.get_nodes()), 4915)
        self.assertEqual(d3plot.get_nElements(), 4696)
        self.assertEqual(d3plot.get_nElements(Element.beam), 0)
        self.assertEqual(d3plot.get_nElements(Element.shell), 4696)
        self.assertEqual(d3plot.get_nElements(Element.solid), 0)
        self.assertEqual(len(d3plot.get_elements()), 4696)
        self.assertEqual(np.sum([len(part.get_elements())
                                 for part in d3plot.get_parts()]), 4696)
        self.assertEqual(np.sum([len(part.get_elements(Element.beam))
                                 for part in d3plot.get_parts()]), 0)
        self.assertEqual(np.sum([len(part.get_elements(Element.shell))
                                 for part in d3plot.get_parts()]), 4696)
        self.assertEqual(np.sum([len(part.get_elements(Element.solid))
                                 for part in d3plot.get_parts()]), 0)
        self.assertEqual(d3plot.get_timesteps()[0], 0.)
        self.assertEqual(len(d3plot.get_timesteps()), 1)
        self.assertEqual(len(d3plot.get_parts()), 1)
        part = d3plot.get_parts()[0]
        self.assertEqual(part.get_nNodes(), 4915)
        self.assertEqual(part.get_nElements(), 4696)
        self.assertEqual(
            part.get_element_node_ids(Element.shell, 4).shape, (4696, 4))
        self.assertEqual(part.get_element_node_indexes(
            Element.shell, 4).shape, (4696, 4))
        self.assertEqual(part.get_node_ids().shape, (4915,))
        self.assertEqual(part.get_node_indexes().shape, (4915,))
        self.assertEqual(part.get_element_ids().shape, (4696,))
        self.assertEqual(part.get_element_ids(Element.beam).shape, (0,))
        self.assertEqual(part.get_element_ids(Element.shell).shape, (4696,))
        self.assertEqual(part.get_element_ids(Element.tshell).shape, (0,))
        self.assertEqual(part.get_element_ids(Element.solid).shape, (0,))
        self.assertEqual(d3plot.get_element_energy().shape, (4696, 1))
        self.assertEqual(d3plot.get_element_energy(Element.beam).shape, (0, 1))
        self.assertEqual(d3plot.get_element_energy(
            Element.shell).shape, (4696, 1))
        self.assertEqual(d3plot.get_element_energy(
            Element.solid).shape, (0, 1))
        self.assertEqual(d3plot.get_element_energy(
            Element.tshell).shape, (0, 1))
        self.assertEqual(d3plot.get_element_plastic_strain().shape, (4696, 1))
        self.assertEqual(d3plot.get_element_plastic_strain(
            Element.beam).shape, (0, 1))
        self.assertEqual(d3plot.get_element_plastic_strain(
            Element.shell).shape, (4696, 1))
        self.assertEqual(d3plot.get_element_plastic_strain(
            Element.solid).shape, (0, 1))
        self.assertEqual(d3plot.get_element_plastic_strain(
            Element.tshell).shape, (0, 1))
        self.assertEqual(d3plot.get_element_stress_mises().shape, (4696, 1))
        self.assertEqual(d3plot.get_element_stress_mises(
            Element.beam).shape, (0, 1))
        self.assertEqual(d3plot.get_element_stress_mises(
            Element.shell).shape, (4696, 1))
        self.assertEqual(d3plot.get_element_stress_mises(
            Element.solid).shape, (0, 1))
        self.assertEqual(d3plot.get_element_stress_mises(
            Element.tshell).shape, (0, 1))
        self.assertEqual(d3plot.get_element_strain().shape, (4696, 1, 6))
        self.assertEqual(d3plot.get_element_strain(
            Element.beam).shape, (0, 1, 6))
        self.assertEqual(d3plot.get_element_strain(
            Element.shell).shape, (4696, 1, 6))
        self.assertEqual(d3plot.get_element_strain(
            Element.solid).shape, (0, 1, 6))
        self.assertEqual(d3plot.get_element_strain(
            Element.tshell).shape, (0, 1, 6))
        self.assertEqual(d3plot.get_element_stress().shape, (4696, 1, 6))
        self.assertEqual(d3plot.get_element_stress(
            Element.beam).shape, (0, 1, 6))
        self.assertEqual(d3plot.get_element_stress(
            Element.shell).shape, (4696, 1, 6))
        self.assertEqual(d3plot.get_element_stress(
            Element.solid).shape, (0, 1, 6))
        self.assertEqual(d3plot.get_element_stress(
            Element.tshell).shape, (0, 1, 6))
        self.assertEqual(d3plot.get_element_coords().shape, (4696, 1, 3))
        self.assertEqual(d3plot.get_element_coords(
            Element.beam).shape, (0, 1, 3))
        self.assertEqual(d3plot.get_element_coords(
            Element.shell).shape, (4696, 1, 3))
        self.assertEqual(d3plot.get_element_coords(
            Element.solid).shape, (0, 1, 3))
        self.assertEqual(d3plot.get_element_coords(
            Element.tshell).shape, (0, 1, 3))
        with self.assertRaises(ValueError):
            d3plot.get_element_history_vars(Element.none)
        with self.assertRaises(ValueError):
            d3plot.get_element_history_vars()
        self.assertEqual(d3plot.get_element_history_vars(
            Element.beam).shape, (0, 1, 0))
        self.assertEqual(d3plot.get_element_history_vars(
            Element.shell).shape, (4696, 1, 1))
        self.assertEqual(d3plot.get_element_history_vars(
            Element.solid).shape, (0, 1, 0))
        self.assertEqual(d3plot.get_element_history_vars(
            Element.tshell).shape, (0, 1, 0))

        # D3plot error handling
        # ... TODO

        # Part
        part1 = d3plot.get_parts()[0]
        self.assertTrue(part1.get_name() == "Zugprobe")
        self.assertTrue(part1.get_id() == 1)
        self.assertTrue(len(part1.get_elements()) == 4696)
        self.assertTrue(len(part1.get_nodes()) == 4915)

        # Node
        node_ids = [1, 2]
        nodes_ids_v1 = [d3plot.get_nodeByID(node_id) for node_id in node_ids]
        nodes_ids_v2 = d3plot.get_nodeByID(node_ids)
        self.assertCountEqual([node.get_id()
                               for node in nodes_ids_v1], node_ids)
        self.assertCountEqual([node.get_id()
                               for node in nodes_ids_v2], node_ids)
        for node in nodes_ids_v1:
            self.assertEqual(node.get_coords().shape, (1, 3))
            self.assertEqual(len(node.get_disp()), 1)
            self.assertEqual(len(node.get_vel()), 1)
            self.assertEqual(len(node.get_accel()), 1)
            self.assertGreater(len(node.get_elements()), 0)

        node_indexes = [0, 1]
        node_matching_ids = [1, 2]  # looked it up manually
        self.assertEqual(len(node_indexes), len(node_matching_ids))
        nodes_indexes_v1 = [d3plot.get_nodeByIndex(
            node_index) for node_index in node_indexes]
        nodes_indexes_v2 = d3plot.get_nodeByIndex(node_indexes)
        self.assertCountEqual([node.get_id()
                               for node in nodes_indexes_v1], node_matching_ids)
        self.assertCountEqual([node.get_id()
                               for node in nodes_indexes_v2], node_matching_ids)
        # .. TODO Error stoff

        # Node Velocity and Acceleration Testing
        self.assertCountEqual(d3plot.get_node_velocity().shape, (4915, 1, 3))
        self.assertCountEqual(
            d3plot.get_node_acceleration().shape, (4915, 1, 3))
        self.assertCountEqual(d3plot.get_node_ids().shape, [4915])

        # Shell Element
        element_ids = [1, 2]
        element_ids_shell_v1 = [d3plot.get_elementByID(
            Element.shell, element_id) for element_id in element_ids]
        element_ids_shell_v2 = d3plot.get_elementByID(
            Element.shell, element_ids)
        self.assertCountEqual([element.get_id()
                               for element in element_ids_shell_v1], element_ids)
        self.assertCountEqual([element.get_id()
                               for element in element_ids_shell_v2], element_ids)
        elem1 = element_ids_shell_v1[0]
        elem2 = element_ids_shell_v2[0]
        for element in element_ids_shell_v1:
            pass
        self.assertEqual(elem1.get_coords().shape, (1, 3))
        self.assertEqual(elem1.get_plastic_strain().shape, (1,))
        self.assertEqual(elem1.get_stress().shape, (1, 6))
        self.assertEqual(elem1.get_stress_mises().shape, (1,))
        self.assertEqual(elem1.get_strain().shape, (1, 6))
        self.assertEqual(elem1.get_history_variables().shape, (1, 1))

        self.assertCountEqual(d3plot.get_element_ids().shape, (4696,))
        self.assertCountEqual(d3plot.get_element_node_ids(
            Element.shell, 4).shape, (4696, 4))

        # .. TODO Error stoff

        # plotting (disabled)
        '''
        export_path = os.path.join(
            os.path.dirname(__file__), "test_export.html")
        for element_result in element_result_list:

            # test d3plot.plot directly
            d3plot.plot(iTimestep=-1,
                        element_result=element_result,
                        export_filepath=export_path)
            self.assertTrue(os.path.isfile(export_path))
            os.remove(export_path)

            # test plotting by parts
            D3plot.plot_parts(d3plot.get_parts(),
                              iTimestep=-1,
                              element_result=element_result,
                              export_filepath=export_path)
            self.assertTrue(os.path.isfile(export_path))
            os.remove(export_path)

            for part in d3plot.get_parts():
                part.plot(iTimestep=-1,
                          element_result=element_result,
                          export_filepath=export_path)
                self.assertTrue(os.path.isfile(export_path))
                os.remove(export_path)

            for part_id in part_ids:
                d3plot.get_partByID(part_id).plot(
                    iTimestep=-1, element_result=None, export_filepath=export_path)
                self.assertTrue(os.path.isfile(export_path))
                os.remove(export_path)
        '''

    def test_binout(self):

        binout_filepath = "test/binout"
        nTimesteps = 321
        content = ["swforc"]  # TODO: special case rwforc
        content_subdirs = [['title', 'failure', 'ids', 'failure_time',
                            'typenames', 'axial', 'version', 'shear', 'time', 'date',
                            'length', 'resultant_moment', 'types', 'revision']]

        # open file
        binout = Binout(binout_filepath)

        # check directory stuff
        self.assertEqual(len(binout.read()), len(content))
        self.assertCountEqual(content, binout.read())

        # check variables reading
        for content_dir, content_subdirs in zip(content, content_subdirs):
            self.assertCountEqual(content_subdirs, binout.read(content_dir))
            self.assertEqual(nTimesteps, len(binout.read(content_dir, "time")))

            for content_subdir in content_subdirs:
                # check if data containers not empty ...
                self.assertGreater(
                    len(binout.read(content_dir, content_subdir)), 0)

        # check string conversion
        self.assertEqual(binout.read("swforc", "typenames"),
                         'constraint,weld,beam,solid,non nodal, ,solid assembly')

        # test saving
        binout.save_hdf5("./binout.h5")
        self.assertTrue(os.path.isfile("./binout.h5"))
        os.remove("./binout.h5")

    def test_keyfile(self):

        # test encryption detection
        np.testing.assert_almost_equal(get_file_entropy(
            "test/keyfile_include2.key"), 7.433761, decimal=6)

        # file loading (arguments)
        kf = KeyFile("test/keyfile.key")
        self.assertEqual(len(kf.keys()), 8)
        self.assertEqual(len(kf.get_includes()), 0)
        self.assertEqual(kf.get_nNodes(), 0)
        self.assertTrue(isinstance(kf["*INCLUDE"][0], Keyword))
        self.assertTrue(isinstance(kf["*NODE"][0], Keyword))
        self.assertTrue(isinstance(kf["*PART"][0], Keyword))
        self.assertTrue(isinstance(kf["*ELEMENT_SHELL"][0], Keyword))

        kf = KeyFile("test/keyfile.key", load_includes=True)
        self.assertEqual(len(kf.keys()), 8)

        self.assertEqual(len(kf.get_includes()), 2)
        self.assertEqual(kf.get_nNodes(), 0)
        self.assertTrue(isinstance(kf["*INCLUDE"][0], IncludeKeyword))
        self.assertTrue(isinstance(kf["*NODE"][0], Keyword))
        self.assertTrue(isinstance(kf["*PART"][0], Keyword))
        self.assertTrue(isinstance(kf["*ELEMENT_SHELL"][0], Keyword))

        kf = KeyFile("test/keyfile.key", load_includes=True, parse_mesh=True)
        self.assertEqual(len(kf.keys()), 8)
        self.assertEqual(len(kf.get_includes()), 2)
        self.assertEqual(kf.get_nNodes(), 6)
        self.assertCountEqual(kf.get_nodeByID(
            11).get_coords()[0], (7., 7., 7.))
        self.assertTrue(isinstance(kf["*INCLUDE"][0], Keyword))
        self.assertTrue(isinstance(kf["*NODE"][0], NodeKeyword))
        self.assertTrue(isinstance(kf["*PART"][0], PartKeyword))
        self.assertTrue(isinstance(kf["*ELEMENT_SHELL"][0], ElementKeyword))

        kf = KeyFile("test/keyfile.key", read_keywords=False)
        self.assertEqual(len(kf.keys()), 0)
        self.assertEqual(len(kf.get_includes()), 0)
        self.assertEqual(kf.get_nNodes(), 0)

        with self.assertRaises(ValueError):
            KeyFile("test/invalid.key")
        with self.assertRaises(ValueError):
            KeyFile("test/invalid.key",
                    parse_mesh=True)

        # saving
        kf = KeyFile("test/keyfile.key")
        self.assertEqual(kf.get_filepath(), "test/keyfile.key")
        kf.save("test/tmp.key")
        self.assertEqual(kf.get_filepath(), "test/tmp.key")
        self.assertTrue(filecmp.cmp("test/keyfile.key", "test/tmp.key"))
        os.remove("test/tmp.key")

        kf = KeyFile("test/keyfile.key", read_keywords=True,
                     parse_mesh=False, load_includes=True)
        _, kf2 = kf.get_includes()
        kf2.save("test/tmp.key")
        self.assertTrue(filecmp.cmp(
            "test/keyfile_include2.key", "test/tmp.key"))
        os.remove("test/tmp.key")

        # Generic Keywords
        kf = KeyFile("test/keyfile.key")
        kwrds = kf["*PART"]
        self.assertEqual(len(kwrds), 1)
        kw = kwrds[0]

        # getter
        self.assertEqual(kw["pid"], 1)
        self.assertEqual(kw.get_card_valueByName("pid"), 1)
        self.assertEqual(kw["secid"], 1)
        self.assertEqual(kw.get_card_valueByName("secid"), 1)
        self.assertEqual(kw["mid"], 1)
        self.assertEqual(kw.get_card_valueByName("mid"), 1)
        self.assertEqual(kw[1, 0], 1)
        self.assertEqual(kw.get_card_valueByIndex(1, 0), 1)
        self.assertEqual(kw[1, 1], 1)
        self.assertEqual(kw.get_card_valueByIndex(1, 1), 1)
        self.assertEqual(kw[1, 2], 1)
        self.assertEqual(kw.get_card_valueByIndex(1, 2), 1)
        # self.assertEqual(kw[1, 0, 50], "1       1") # TODO fix this
        self.assertEqual(kw[0, 0], "Iam beauti")
        self.assertEqual(kw.get_card_valueByIndex(0, 0), "Iam beauti")
        self.assertEqual(kw[0, 0, 80], "Iam beautiful")
        self.assertEqual(kw.get_card_valueByIndex(0, 0, 80), "Iam beautiful")
        with self.assertRaises(ValueError):
            kw["error"]
        with self.assertRaises(ValueError):
            kw.get_card_valueByName("error")
        with self.assertRaises(ValueError):
            kw.get_card_valueByIndex(100, 0)

        # setter
        kw["pid"] = 100
        self.assertEqual(kw["pid"], 100)
        kw.set_card_valueByName("pid", 200)
        self.assertEqual(kw["pid"], 200)
        kw["pid"] = 12345678912
        self.assertEqual(kw["pid"], 1234567891)
        kw.set_card_valueByName("pid", 12345678912)
        self.assertEqual(kw["pid"], 1234567891)
        self.assertEqual(kw["secid"], 1)

        kw[1, 0] = 300
        self.assertEqual(kw[1, 0], 300)
        kw.set_card_valueByIndex(1, 0, 400)
        self.assertEqual(kw[1, 0], 400)
        kw[1, 0] = 12345678912
        self.assertEqual(kw[1, 0], 1234567891)
        self.assertEqual(kw[1, 1], 1)
        kw[0, 0, 30] = "Hihihi   "
        self.assertEqual(kw[0, 0, 30], "Hihihi")
        kw.set_card_valueByDict({(1, 0): "yoy", "mid": "ok"})
        self.assertEqual(kw[1, 0], "yoy")
        self.assertEqual(kw["mid"], "ok")

        # line manipulation
        line_data = ["*PART",
                     "$ heading",
                     "Iam beautiful",
                     "$    pid      secid       mid",
                     "       1          1         1",
                     "       "]

        kf = KeyFile("test/keyfile.key")
        kw = kf["*PART"][0]

        self.assertEqual(len(kw), len(line_data))
        self.assertCountEqual(kw.get_lines(), line_data)
        for iLine, line in enumerate(line_data):
            self.assertEqual(kw.get_line(iLine), line)

        kw.insert_line(1, "$ another comment")
        self.assertEqual(kw.get_line(1), "$ another comment")

        kw.append_line("$ changed comment")
        self.assertEqual(kw.get_line(7), "$ changed comment")

        kw.remove_line(7)
        kw.remove_line(1)
        self.assertEqual(len(kw), len(line_data))

        kw.append_line("blubber")
        kw.set_lines(line_data)
        self.assertCountEqual(kw.get_lines(), line_data)

        # str
        kf = KeyFile("test/keyfile.key")
        kw = kf["*PART"][0]
        kw_data = '*PART\n$ heading\nIam beautiful\n$    pid      secid       mid\n       1          1         1\n       \n'
        self.assertEqual(str(kw), kw_data)

        # reformatting
        Keyword.name_delimiter_used = True
        Keyword.name_delimiter = '|'
        Keyword.name_spacer = '-'
        Keyword.name_alignment = Keyword.align.right
        Keyword.field_alignment = Keyword.align.right

        kw.reformat_all([0])
        kw_data = '*PART\n$ heading\nIam beautiful\n$------pid|----secid|------mid\n         1         1         1\n          \n'
        self.assertEqual(str(kw), kw_data)

        kw.reformat_field(0, 0, 80)
        kw_data = '*PART\n$------------------------------------------------------------------------heading\n                                                                   Iam beautiful\n$------pid|----secid|------mid\n         1         1         1\n          \n'
        self.assertEqual(str(kw), kw_data)

        # NodeKeyword
        kf = KeyFile("test/keyfile.key", load_includes=True, parse_mesh=True)
        kw = kf["*NODE"][0]
        self.assertEqual(kw.get_nNodes(), 4)
        self.assertEqual(len(kw.get_nodes()), 4)
        self.assertCountEqual(kw.get_node_ids(), [1, 3, 4, 5])
        node = kw.add_node(6, 4., 4., 4., "       0       0")
        self.assertEqual(node.get_id(), 6)
        np.testing.assert_array_almost_equal(node.get_coords()[0],
                                             [4., 4., 4.],
                                             decimal=3)

        node_kw_data = "*NODE\n$ some comment line\n$     id              x               y               z\n       1               0               0               0       0       0\n       3               0               1               0       0       0\n       4               0               1               1       0       0\n       5               1               0               0       0       0\n       6               4               4               4       0       0\n\n"
        self.assertEqual(str(kw), node_kw_data)
        kw.append_line(
            "       7              8.              8.              8.       0       0")
        kw.load()
        self.assertEqual(kw.get_nNodes(), 6)
        node = kf.get_nodeByID(7)
        np.testing.assert_array_almost_equal(node.get_coords()[0],
                                             [8., 8., 8.],
                                             decimal=3)

        # PartKeyword
        kf = KeyFile("test/keyfile.key", load_includes=True, parse_mesh=True)
        kw = kf["*PART"][0]
        self.assertEqual(kw.get_nParts(), 1)
        self.assertEqual(len(kw.get_parts()), 1)
        part = kw.get_parts()[0]
        self.assertEqual(part.get_name(), "Iam beautiful")
        self.assertEqual(part.get_id(), 1)
        part = kw.add_part(2, "new part", "        1         1")

        part_data = "*PART\n$ heading\n                                                         Iam beautiful\n$    pid      secid       mid\n         1        1         1\n                                                              new part\n         2        1         1\n"
        self.assertEqual(str(kw), part_data)
        self.assertEqual(kw.get_nParts(), 2)
        self.assertEqual(kw.get_parts()[1].get_name(), "new part")

        # ElementKeyword
        kf = KeyFile("test/keyfile.key", load_includes=True, parse_mesh=True)
        kw = kf["*ELEMENT_SHELL"][0]
        self.assertEqual(kw.get_nElements(), 1)
        self.assertEqual(len(kw.get_elements()), 1)

        elem = kw.get_elements()[0]
        self.assertEqual(elem.get_type(), Element.shell)
        self.assertEqual(elem.get_id(), 1)

        elem = kw.add_elementByNodeID(2, 1, [1, 2, 3, 4])
        elem_data = "*ELEMENT_SHELL\n$    eid     pid      n1      n2      n3      n4      n5      n6      n7      n8\n       1       1       1       2       3       4       1       1       1       1\n       2       1       1       2       3       4\n\n"
        self.assertEqual(str(kw), elem_data)
        self.assertEqual(elem.get_type(), Element.shell)
        self.assertCountEqual([node.get_id()
                               for node in elem.get_nodes()], [1, 2, 3, 4])

        # ElementKeyword new solid element format test
        kf = KeyFile(parse_mesh=True)
        nkw = kf.add_keyword("*NODE")
        nkw.add_node(42731096, 0, 0, 0)
        nkw.add_node(42730740, 0, 0, 0)
        nkw.add_node(42731090, 0, 0, 0)
        nkw.add_node(42723028, 0, 0, 0)

        pkw = kf.add_keyword("*PART")
        pkw.add_part(42792001, "yay")

        ekw = kf.add_keyword("*ELEMENT_SOLID")
        ekw.append_line("4278837942792001")
        ekw.append_line(
            "4273109642730740427310904272302842723028427230284272302842723028       0       0")
        ekw.load()

        self.assertEqual(len(kf.get_elements()), 1)
        elem = kf.get_elements()[0]
        self.assertEqual(elem.get_id(), 42788379)
        self.assertEqual(elem.get_part_id(), 42792001)
        self.assertEqual(len(elem.get_nodes()), 4)
        self.assertEqual(elem.get_type(), Element.solid)

        # IncludePathKeyword
        kf = KeyFile("test/keyfile.key", load_includes=True)

        kw = kf.add_keyword("*INCLUDE_PATH_RELATIVE")
        kw.append_line("subdir")
        self.assertCountEqual(kw.get_include_dirs(), ["subdir"])
        self.assertCountEqual(
            kw.get_lines(), ["*INCLUDE_PATH_RELATIVE", "subdir"])

        kw = kf.add_keyword("*INCLUDE_PATH")
        kw.append_line("C:/absolute/path")
        self.assertCountEqual(kw.get_include_dirs(), ["C:/absolute/path"])
        self.assertCountEqual(
            kw.get_lines(), ["*INCLUDE_PATH", "C:/absolute/path"])

        # self.assertEqual(kf.get_include_dirs(), [
        #                  'C:/absolute/path', 'test/', 'test/subdir'])
        self.assertEqual(kf.get_include_dirs(), [
                         'C:/absolute/path', 'subdir', 'test/',
                         'test/C:/absolute/path', 'test/subdir'])

        # IncludeKeyword
        kf = KeyFile("test/keyfile.key", load_includes=True)

        kw = kf.add_keyword("*INCLUDE_PATH_RELATIVE")
        kw.append_line("keyfile_include_dir")

        kw = kf["*INCLUDE"][0]
        self.assertEqual(len(kw.get_includes()), 1)
        inc_kf = kw.get_includes()[0]
        self.assertEqual(len(inc_kf.keys()), 3)

        kw.append_line("keyfile_include3.key")
        kw.load()
        self.assertEqual(len(kw.get_includes()), 2)
        self.assertEqual(len(kw.get_includes()[1].keys()), 3)

        # test mesh
        kf = KeyFile("test/keyfile.key", load_includes=True, parse_mesh=True)
        node_coords = np.array(
            [[0., 0., 0.], [2., 2., 2.], [0., 1., 0.], [0., 1., 1.], [1., 0., 0.]])
        nodes = kf.get_nodeByID([1, 2, 3, 4, 5])
        for iNode, node in enumerate(nodes):
            np.testing.assert_array_almost_equal(node.get_coords()[0],
                                                 node_coords[iNode],
                                                 decimal=3)

        elem_nodes = np.array([[1, 2, 3, 4], [1, 2]])
        elems = [(Element.shell, 1), (Element.beam, 1)]
        for iElem, (etype, eid) in enumerate(elems):
            self.assertCountEqual(elem_nodes[iElem],
                                  [node.get_id()
                                   for node in kf.get_elementByID(etype, eid).get_nodes()])

        part = kf.get_partByID(1)
        self.assertEqual(part.get_name(), "Iam beautiful")
        self.assertEqual(len(part.get_elements()), 2)
        self.assertEqual(len(part.get_nodes()), 4)

    def test_keyfile_issue_66(self):

        kf = KeyFile(parse_mesh=True)
        kw = kf.add_keyword("*Part")

        additional_data = [" 2000001 2000017"]
        # additional_data = " 2000001 2000017"
        for pid in range(19, 22):
            kw.add_part(pid, "A", additional_lines=additional_data)

        part_data = "\n".join([
            "*Part",
            "                                                                     A",
            "        19 2000001 2000017",
            "                                                                     A",
            "        20 2000001 2000017",
            "                                                                     A",
            "        21 2000001 2000017\n",
        ])

        self.assertEqual(str(kw), part_data)

    def test_raw_d3plot(self):

        d3plot_filepath = "test/d3plot"

        keys_data = {'part_names': ['Zugprobe                                                                '],
                     'elem_shell_results': (1, 4696, 24), 'node_ids': (4915,),
                     'timesteps': (1,), 'node_coordinates': (4915, 3), 'node_acceleration': (1, 4915, 3),
                     'elem_shell_data': (4696, 5), 'elem_shell_ids': (4696,),
                     'node_displacement': (1, 4915, 3),  'part_ids': (1,),
                     'node_velocity': (1, 4915, 3), 'elem_shell_results_layers': (1, 4696, 3, 26),
                     'elem_shell_deletion_info': (1, 4696)}

        keys_names = sorted(keys_data.keys())

        # do the thing
        raw_d3plot = RawD3plot(d3plot_filepath)

        # test
        self.assertEqual(sorted(raw_d3plot.get_raw_keys()), keys_names)
        for key in raw_d3plot.get_raw_keys():
            data = raw_d3plot.get_raw_data(key)
            if isinstance(data, np.ndarray):
                self.assertEqual(data.shape, keys_data[key])
            elif isinstance(data, str):
                self.assertEqual(data, keys_data[key])

        # saving hdf5
        raw_d3plot.save_hdf5("./test.h5")
        self.assertTrue(os.path.isfile("./test.h5"))

        # reading hdf5
        raw_d3plot = RawD3plot("./test.h5")
        os.remove("./test.h5")
        self.assertEqual(sorted(raw_d3plot.get_raw_keys()), keys_names)
        for key in raw_d3plot.get_raw_keys():
            data = raw_d3plot.get_raw_data(key)
            if isinstance(data, np.ndarray):
                self.assertEqual(data.shape, keys_data[key])
            elif isinstance(data, str):
                self.assertEqual(data, keys_data[key])

    def test_numerics_sampling(self):
        '''Testing qd.numerics'''

        from qd.numerics.sampling import uniform_lhs

        nSamples = 100
        vars = {"a": [0, 5], "b": [-10, 10], "c": [0, 1]}
        var_labels, samples = uniform_lhs(nSamples, vars)

        assert len(vars) == len(var_labels)
        assert all(label in vars for label in var_labels)
        assert samples.shape[0] == nSamples
        assert samples.shape[1] == len(vars)
        assert np.amin(samples[:, var_labels.index("a")]) >= 0
        assert np.amax(samples[:, var_labels.index("a")]) <= 5
        assert np.amin(samples[:, var_labels.index("b")]) >= -10
        assert np.amax(samples[:, var_labels.index("b")]) <= 10
        assert np.amin(samples[:, var_labels.index("c")]) >= 0
        assert np.amax(samples[:, var_labels.index("c")]) <= 1

    def test_qd_cae_beta(self):
        '''Testing qd.cae.beta'''

        from qd.cae.beta import MetaCommunicator
        # ...


if __name__ == "__main__":
    pass
