.. currentmodule:: libdvid._dvid_python

DVIDNodeService
===============

.. _DVIDNodeService:

.. autoclass:: DVIDNodeService

  **Methods List:**

  - :py:meth:`__init__ <DVIDNodeService.__init__>`

  - Key-value

    - :py:meth:`create_keyvalue <DVIDNodeService.create_keyvalue>`
    - :py:meth:`put <DVIDNodeService.put>`
    - :py:meth:`get <DVIDNodeService.get>`
    - :py:meth:`get_json <DVIDNodeService.get_json>`
    - :py:meth:`get_keys <DVIDNodeService.get_keys>`

  - Grayscale

    - :py:meth:`create_grayscale8 <DVIDNodeService.create_grayscale8>`
    - :py:meth:`get_gray3D <DVIDNodeService.get_gray3D>`
    - :py:meth:`put_gray3D <DVIDNodeService.put_gray3D>`

  - Labels

    - :py:meth:`create_labelblk <DVIDNodeService.create_labelblk>`
    - :py:meth:`put_labels3D <DVIDNodeService.put_labels3D>`
    - :py:meth:`get_labels3D <DVIDNodeService.get_labels3D>`
    - :py:meth:`get_label_by_location <DVIDNodeService.get_label_by_location>`
    - :py:meth:`body_exists <DVIDNodeService.body_exists>`

  - Roi

    - :py:meth:`create_roi <DVIDNodeService.create_roi>`
    - :py:meth:`get_roi <DVIDNodeService.get_roi>`
    - :py:meth:`post_roi <DVIDNodeService.post_roi>`
    - :py:meth:`get_roi_partition <DVIDNodeService.get_roi_partition>`
    - :py:meth:`roi_ptquery <DVIDNodeService.roi_ptquery>`
    - :py:meth:`get_roi3D <DVIDNodeService.get_roi3D>`

  - Tiles

    - :py:meth:`get_tile_slice <DVIDNodeService.get_tile_slice>`
    - :py:meth:`get_tile_slice_binary <DVIDNodeService.get_tile_slice_binary>`

  - Graph

    - :py:meth:`create_graph <DVIDNodeService.create_graph>`
    - :py:meth:`update_vertices <DVIDNodeService.update_vertices>`
    - :py:meth:`update_edges <DVIDNodeService.update_edges>`

  - General

    - :py:meth:`get_typeinfo <DVIDNodeService.get_typeinfo>`
    - :py:meth:`custom_request <DVIDNodeService.custom_request>`

  **Methods Reference:**

   .. automethod:: __init__
   
   .. Key-value
   
   .. automethod:: create_keyvalue   
   .. automethod:: put
   .. automethod:: get
   .. automethod:: get_json
   .. automethod:: get_keys

   .. Grayscale

   .. automethod:: create_grayscale8
   .. automethod:: get_gray3D
   .. automethod:: put_gray3D

   .. Labels

   .. automethod:: create_labelblk
   .. automethod:: get_labels3D
   .. automethod:: get_label_by_location
   .. automethod:: put_labels3D
   .. automethod:: body_exists

   .. Roi

   .. automethod:: create_roi
   .. automethod:: get_roi
   .. automethod:: post_roi
   .. automethod:: get_roi_partition
   .. automethod:: roi_ptquery
   .. automethod:: get_roi3D

   .. Tiles
   
   .. automethod:: get_tile_slice
   .. automethod:: get_tile_slice_binary

   .. Graph

   .. automethod:: create_graph
   .. automethod:: update_vertices
   .. automethod:: update_edges

   .. General

   .. automethod:: get_typeinfo
   .. automethod:: custom_request


Slice2D
-------

.. autoclass:: Slice2D

Graph Primitives
----------------

.. autoclass:: Vertex

   .. automethod:: __init__

.. autoclass:: Edge

   .. automethod:: __init__
