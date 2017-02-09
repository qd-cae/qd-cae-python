
import os
import io
import json
import uuid
import numpy as np
from base64 import b64encode
from zipfile import ZipFile, ZIP_DEFLATED

from .dyna import Part


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


def _extract_elem_data(element, iTimestep=0, result_type=None):
    '''Extract fringe data from the element

    Parameters:
    -----------
    element : Element
        Element to extract the data of
    iTimestep : int
        timestep at which to extract the element data
    result_type : str
        type of result data to extract

    Returns:
    --------
    result : float

    Returns None in case that no result type is desired.
    '''

    if result_type == None:
        return None
    elif "plastic_strain" in result_type:
        return element.get_plastic_strain()[iTimestep]
    elif "energy" in result_type:
        return element.get_energy()[iTimestep]
    else:
        raise ValueError("Unknown result type %s. Try energy or plastic_strain." % result_type)


def _extract_mesh_from_parts(parts, iTimestep=0, result_type=None):
    '''Extract the mesh data from a part or list of parts

    Parameters:
    -----------
    parts : part or list(part)
        parts to extract mesh of
    iTimestep : int
        timestep at which to extract the mesh coords
    result_type : str
        type of results to read

    Returns:
    --------
        mesh_coords, mesh_fringe, elem_texts : tuple(np.ndarray, np.ndarray, list)
    '''

    if isinstance(parts, Part):
        parts = [parts]

    node_data   = []
    node_fringe = []
    element_texts = []

    # loop through parts elements
    for part in parts:
        for elem in part.get_elements():

            if elem.get_type() != "shell":
                continue
            
            elem_nodes = elem.get_nodes()

            # element fringe
            elem_result = _extract_elem_data(elem, iTimestep, result_type)

            # element annotation
            if elem_result == None:
                elem_result = 0.
                element_texts.append("e#%d" % elem.get_id())
            else:
                element_texts.append("e#%d=%.5f" % (elem.get_id(), elem_result) )
            
            # extract nodal data
            for node in elem_nodes[:3]:
                node_data.append( node.get_coords(iTimestep) )
                node_fringe.append(elem_result)

            # add second tria for quad shells
            if len(elem_nodes) == 4: 
                element_texts.append(None)
                node_data.append( elem_nodes[0].get_coords(iTimestep) )
                node_data.append( elem_nodes[2].get_coords(iTimestep) )
                node_data.append( elem_nodes[3].get_coords(iTimestep) )
                node_fringe.append(elem_result)
                node_fringe.append(elem_result)
                node_fringe.append(elem_result)

    # wrap up data
    node_data = np.asarray(node_data, dtype=np.float32).flatten()
    node_fringe = np.asarray(node_fringe, dtype=np.float32)
    
    return node_data, node_fringe, element_texts


def _parts_to_html(parts, iTimestep=0, result_type=None, fringe_bounds=[None,None]):
    '''Convert a part or multiple parts to a 3D HTML

    Parameters:
    -----------
    parts : Part or list(Part)
        part to convert (from a d3plot)
    iTimestep : int
        timestep at which the coordinates are taken
    result_type : str
        type of results to read, None means no fringe
    fringe_bounds : list(float,float) or tuple(float,float)
        bounds for the fringe, default will use min and max value

    Returns:
    --------
    html : str
        the 3D HTML as string
    '''
    
    # extract mesh dta
    node_coords, node_fringe, element_texts = _extract_mesh_from_parts(parts, iTimestep=iTimestep, result_type=result_type)

    # fringe bounds
    fringe_bounds = list(fringe_bounds) # convert in case of tuple (no assignments)
    fringe_bounds[0] = np.amin(node_fringe) if fringe_bounds[0] == None else fringe_bounds[0]
    fringe_bounds[1] = np.amax(node_fringe) if fringe_bounds[1] == None else fringe_bounds[1]

    # zip compression of data for HTML (reduces size)
    zdata = io.BytesIO()
    with ZipFile(zdata,'w',compression=ZIP_DEFLATED) as zipFile:
        zipFile.writestr('/intensities', node_fringe.tostring() )
        zipFile.writestr('/positions',   node_coords.tostring() )
        zipFile.writestr('/text', json.dumps(element_texts) )
    zdata = b64encode( zdata.getvalue() ).decode('utf-8')

    # read html template
    _html_template = _read_file(os.path.join(os.path.dirname(__file__),'resources','html.template') )

    # format html template file
    _html_div = _html_template.format(div_id   = uuid.uuid4(),
                                  lowIntensity = fringe_bounds[0],
                                  highIntensity= fringe_bounds[1],
                                  zdata        = zdata)

    # wrap it up with all needed js libraries
    _html_jszip_js  = '<script type="text/javascript">%s</script>' % _read_file(os.path.join(os.path.dirname(__file__),'resources','jszip.min.js') ) 
    _html_three_js  = '<script type="text/javascript">%s</script>' % _read_file(os.path.join(os.path.dirname(__file__),'resources','three.min.js') )  
    _html_chroma_js = '<script type="text/javascript">%s</script>' % _read_file(os.path.join(os.path.dirname(__file__),'resources','chroma.min.js') )
    _html_jquery_js = '<script type="text/javascript">%s</script>' % _read_file(os.path.join(os.path.dirname(__file__),'resources','jquery.min.js') )

    return   '''
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
    _html_div = _html_div,
    _jszip_js = _html_jszip_js,
    _three_js = _html_three_js,
    _chroma_js= _html_chroma_js,
    _jquery_js= _html_jquery_js)

