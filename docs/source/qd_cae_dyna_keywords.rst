

Keyword Types
-------------

The basic idea of these classes is, that the file should look identical if written again, except for modifications. Also comments shall be preserved. Please beware in case of 

**Keyword**

    The generic `Keyword` class is used for all unknown keywords. If the mesh of a `KeyFile` is not parsed, then also the mesh keywords are seen as generic keywords. To read all keywords of a `KeyFile` use the `read_generic_keywords` option.

    If a `KeyFile` is written again, it will look identical, since the class saves saves everything including the comments.

    .. toctree::
        :maxdepth: 3

        qd_cae_dyna_keywords_Keyword

**Include Keywords**

    The specific include keyword classes are only used, if includes are loaded in the `KeyFile` contructor (`load_includes`). They `IncludePathKeyword` class manages all locations, where include files could be located, while the `IncludeKeyword` is resposible for managing one or multiple includes. 

    .. toctree::
        :maxdepth: 3

        qd_cae_dyna_keywords_IncludeKeyword
        qd_cae_dyna_keywords_IncludePathKeyword

    .. note::
        All the keywords of the includes are not accessible from the main `KeyFile`, but only from the include `KeyFile`, which can be retrieved by `KeyFile.get_includes`.

**Mesh Keywords**
    
    The mesh pecific Keywords:

    .. toctree::
        :maxdepth: 3

        qd_cae_dyna_keywords_NodeKeyword
        qd_cae_dyna_keywords_ElementKeyword
        qd_cae_dyna_keywords_PartKeyword

    are only created, if opening a `KeyFile` with the argument `parse_mesh` enabled. If the mesh is not parsed, their data is seen as a generic `Keyword`.

    .. warning::
        If parsing the mesh, all keywords, except for the `PartKeyword` stop parsing, if they encounter a comment in the data block (e.g. between two elements)

    .. warning::
        Mesh entities, such as nodes can be created, but not deleted. Also these mesh-specific keywords can not be deleted.



