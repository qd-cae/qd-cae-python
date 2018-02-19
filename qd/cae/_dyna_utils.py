
import os
import io
import json
import uuid
import numbers
import tempfile
import webbrowser
import numpy as np
from base64 import b64encode
from zipfile import ZipFile, ZIP_DEFLATED

#from .dyna_cpp import Element, QD_Part
from .dyna_cpp import *


def _read_file(filepath):
    '''This function reads file as str

    Parameters
    ----------
    filename : str
        filepath of the file to read as string

    Returns
    -------
    file_content : str
    '''

    with open(filepath, "r") as fp:
        return fp.read()


def _parse_element_result(arg, iTimestep=-1):
    '''Get the element result function from the argument

    Parameter
    ---------
    arg : str or function(elem)
        type of element result. If it is a function, itself
        is returned. If it is a string, it is converted into
        an evaluation function.
    iTimestep : int
        timestep at which to extract the results

    Returns
    -------
    d3plot_load_var : str
        string for loading variable in a d3plot. None if arg
        is already a function.
    evaluation_function : function(elem)
        evaluation function which takes an element as argument
    '''

    assert isinstance(arg, str) or callable(arg)

    if callable(arg):
        return None, arg

    if arg == "plastic_strain":

        def eval_function(elem):
            _etype = elem.get_type()
            if _etype == Element.shell or _etype == Element.solid:
                return elem.get_plastic_strain()[iTimestep]
            else:
                return 0.

        return "plastic_strain max", eval_function

    elif arg == "energy":

        def eval_function(elem):
            _etype = elem.get_type()
            if _etype == Element.shell or _etype == Element.solid:
                return elem.get_energy()[iTimestep]
            else:
                return 0.

        return "energy", eval_function

    elif arg == "disp":
        return "disp", lambda elem: elem.get_disp(iTimestep)
    else:
        raise ValueError(
            "Unknown result type: %s, Try plastic_strain, energy or disp." % arg)


def _extract_elem_coords(parts, element_result=None, iTimestep=0, element_type=None):
    '''Extract the coordinates of the elements

    Parameters
    ----------
    parts : list(Part)
        list of parts of which to extract the coordinates
    element_result : function
        element evaluation function
    iTimestep : int
        timestep at which to take the coordinates
    element_type : Element.type
        element filter type, beam, shell or solid.

    Returns
    -------
    elem_coords : np.ndarray
        coordinates of the elements
    '''

    # checks
    assert all(isinstance(entry, QD_Part) for entry in parts)
    assert callable(element_result) or element_result == None

    # handle possible results
    if element_result:
        var, eval_function = _parse_element_result(
            element_result, iTimestep=iTimestep)

        def eval_elem(_elem):
            coords.append(_elem.get_coords()[iTimestep])
            elem_results.append(eval_function(_elem))
    else:
        def eval_elem(_elem):
            coords.append(_elem.get_coords())

    # extract
    coords = []
    elem_results = []
    for _part in parts:
        for _elem in _part.get_elements(element_type):
            eval_elem(_elem)
    coords = np.array(coords)
    elem_results = np.array(elem_results)

    # return
    if element_result:
        return coords, elem_results
    else:
        return coords


def _extract_elem_data(element, iTimestep=0, element_result=None):
    '''Extract fringe data from the element

    Parameters
    ----------
    element : Element
        Element to extract the data of
    iTimestep : int
        timestep at which to extract the element data
    element_result : str or function
        which type of results to use as fringe
        None means no fringe is used
        Function shall take elem as input and return a float value (for fringe)

    Returns
    -------
    result : float

    Returns None in case that no result type is desired.
    '''

    if element_result == None:
        return None

    elif isinstance(element_result, str):

        if "plastic_strain" in element_result:
            return element.get_plastic_strain()[iTimestep]
        elif "energy" in element_result:
            return element.get_energy()[iTimestep]
        else:
            raise ValueError(
                "Unknown result type %s. Try energy or plastic_strain." % element_result)

    elif callable(element_result):
        elem_result = element_result(element)
        assert isinstance(
            elem_result, numbers.Number), "The return from the element_result function must be a number!"
        return elem_result

    else:
        raise ValueError(
            "Unkown argument type %s for _extract_elem_data." % str(element_result))


def _extract_mesh_from_parts(parts, iTimestep=0, element_result=None):
    '''Extract the mesh data from a part or list of parts

    Parameters
    ----------
    parts : part or list(part)
        parts to extract mesh of
    iTimestep : int
        timestep at which to extract the mesh coords
    element_result : str or function
        which type of results to use as fringe
        None means no fringe is used
        Function shall take elem as input and return a float value (for fringe)

    Returns
    -------
        mesh_coords, mesh_fringe, elem_texts : tuple(np.ndarray, np.ndarray, list)
    '''

    if isinstance(parts, QD_Part):
        parts = [parts]

    node_data = []
    node_fringe = []
    element_texts = []

    # loop through parts elements
    for part in parts:
        for elem in part.get_elements():

            if elem.get_type() != Element.shell:
                continue

            elem_nodes = elem.get_nodes()

            # element fringe
            elem_result = _extract_elem_data(elem, iTimestep, element_result)

            # element annotation
            if elem_result == None:
                elem_result = 0.
                element_texts.append("e#%d" % elem.get_id())
            else:
                element_texts.append("e#%d=%.5f" %
                                     (elem.get_id(), elem_result))

            # extract nodal data
            for node in elem_nodes[:3]:
                node_data.append(node.get_coords()[iTimestep])
                node_fringe.append(elem_result)

            # add second tria for quad shells
            if len(elem_nodes) == 4:
                element_texts.append(None)
                node_data.append(elem_nodes[0].get_coords()[iTimestep])
                node_data.append(elem_nodes[2].get_coords()[iTimestep])
                node_data.append(elem_nodes[3].get_coords()[iTimestep])
                node_fringe.append(elem_result)
                node_fringe.append(elem_result)
                node_fringe.append(elem_result)

    # wrap up data
    node_data = np.asarray(node_data, dtype=np.float32).flatten()
    node_fringe = np.asarray(node_fringe, dtype=np.float32)

    return node_data, node_fringe, element_texts


def _parts_to_html(parts, iTimestep=0, element_result=None, fringe_bounds=[None, None]):
    '''Convert a part or multiple parts to a 3D HTML

    Parameters
    ----------
    parts : Part or list(Part)
        part to convert (from a d3plot)
    iTimestep : int
        timestep at which the coordinates are taken
    element_result : str or function
        which type of results to use as fringe
        None means no fringe is used
        Function shall take elem as input and return a float value (for fringe)
    fringe_bounds : list(float,float) or tuple(float,float)
        bounds for the fringe, default will use min and max value

    Returns
    -------
    html : str
        the 3D HTML as string
    '''

    if isinstance(parts, QD_Part):
        parts = [parts]

    # extract mesh dta
    node_coords, node_fringe, element_texts = _extract_mesh_from_parts(
        parts, iTimestep=iTimestep, element_result=element_result)

    # fringe bounds
    # convert in case of tuple (no assignments)
    fringe_bounds = list(fringe_bounds)
    fringe_bounds[0] = np.amin(
        node_fringe) if fringe_bounds[0] == None else fringe_bounds[0]
    fringe_bounds[1] = np.amax(
        node_fringe) if fringe_bounds[1] == None else fringe_bounds[1]

    # zip compression of data for HTML (reduces size)
    zdata = io.BytesIO()
    with ZipFile(zdata, 'w', compression=ZIP_DEFLATED) as zipFile:
        zipFile.writestr('/intensities', node_fringe.tostring())
        zipFile.writestr('/positions',   node_coords.tostring())
        zipFile.writestr('/text', json.dumps(element_texts))
    zdata = b64encode(zdata.getvalue()).decode('utf-8')

    # read html template
    _html_template = _read_file(os.path.join(
        os.path.dirname(__file__), 'resources', 'html.template'))

    # format html template file
    _html_div = _html_template.format(div_id=uuid.uuid4(),
                                      lowIntensity=fringe_bounds[0],
                                      highIntensity=fringe_bounds[1],
                                      zdata=zdata)

    # wrap it up with all needed js libraries
    _html_jszip_js = '<script type="text/javascript">%s</script>' % _read_file(
        os.path.join(os.path.dirname(__file__), 'resources', 'jszip.min.js'))
    _html_three_js = '<script type="text/javascript">%s</script>' % _read_file(
        os.path.join(os.path.dirname(__file__), 'resources', 'three.min.js'))
    _html_chroma_js = '<script type="text/javascript">%s</script>' % _read_file(
        os.path.join(os.path.dirname(__file__), 'resources', 'chroma.min.js'))
    _html_jquery_js = '<script type="text/javascript">%s</script>' % _read_file(
        os.path.join(os.path.dirname(__file__), 'resources', 'jquery.min.js'))

    return '''
<!DOCTYPE html>
<html lang="en">
    <head>
    <meta charset="utf-8" />
        {_jquery_js}
        {_jszip_js}
        {_three_js}
        {_chroma_js}
    </head>
    <body>
        {_html_div}
    </body>
</html>'''.format(
        _html_div=_html_div,
        _jszip_js=_html_jszip_js,
        _three_js=_html_three_js,
        _chroma_js=_html_chroma_js,
        _jquery_js=_html_jquery_js)


def _plot_html(_html, export_filepath=None):
    '''Plot a 3D html

    Parameters
    ----------
    _html : str
        html as string
    export_filepath : str
        optional filepath for saving

    This function takes an html as string and if no export_filepath
    is given, directly plots it with your current webbrowser. If 
    a filepath is given, the html will be saved to the given location.
    '''

    # save if export path present
    if export_filepath:
        with open(export_filepath, "w") as fp:
            fp.write(_html)

    # plot if no export
    else:

        # clean temporary dir first (keeps mem low)
        tempdir = tempfile.gettempdir()
        tempdir = os.path.join(tempdir, "qd_eng")
        if not os.path.isdir(tempdir):
            os.mkdir(tempdir)

        for tmpfile in os.listdir(tempdir):
            tmpfile = os.path.join(tempdir, tmpfile)
            if os.path.isfile(tmpfile):
                os.remove(tmpfile)

        # create new temp file
        with tempfile.NamedTemporaryFile(dir=tempdir, suffix=".html", mode="w", delete=False) as fp:
            fp.write(_html)
            webbrowser.open(fp.name)


def plot_parts(parts, iTimestep=0, element_result=None, fringe_bounds=[None, None], export_filepath=None):
    '''Plot a single or multiple parts from a d3plot

    Parameters
    ----------
    parts : Part or list(Part)
        part to convert (from a d3plot)
    iTimestep : int
        timestep at which the coordinates are taken
    element_result : str or function
        None means no fringe is used
        str -> type of results to use as fringe (plastic_strain or energy)
        function -> take elem as input and return a float value (for fringe)
    fringe_bounds : list(float,float) or tuple(float,float)
        bounds for the fringe, default will use min and max value
    export_filepath : str
        optional filepath for saving. If none, the model
        is exported to a temporary file and shown in the
        browser.
    '''

    _html = _parts_to_html(parts,
                           iTimestep=iTimestep,
                           element_result=element_result,
                           fringe_bounds=fringe_bounds)

    _plot_html(_html, export_filepath=export_filepath)


def _extract_surface_mesh(parts):
    '''Extract the surface mesh from parts

    Parameters
    ----------
    parts : Part or list(Part)
        parts from which to extract the mesh

    Returns
    -------
    ? : ?
    '''
    if not isinstance(parts, (list, tuple, np.ndarray)):
        parts = [parts]
    assert all(isinstance(Part, entry) for entry in parts)

    faces = {}
    for _part in parts:
        for _element in _part.get_elements(Element.solid):

            # get neighbor elems nodes (complicated ...)
            neighbor_elems_nodes = []
            nodes = _element.get_nodes()
            for _node in nodes:
                neighbor_elems += _node.get_elements()
