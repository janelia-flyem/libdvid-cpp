"""
This module implements a simple widget for viewing the list of datasets and nodes in a DVID instance.
Requires PyQt4.  To see a demo of it in action, start up your dvid server run this::

$ python contents_browser.py localhost:8000
"""
import json
import httplib
import collections

from PyQt4.QtCore import Qt, QStringList, QSize, QEvent
from PyQt4.QtGui import QPushButton, QDialog, QVBoxLayout, QGroupBox, QTreeWidget, \
                        QTreeWidgetItem, QSizePolicy, QListWidget, QListWidgetItem, \
                        QDialogButtonBox, QLineEdit, QLabel, QComboBox, QMessageBox, \
                        QHBoxLayout, QTableWidget, QTableWidgetItem

from libdvid import DVIDConnection, ConnectionMethod, ErrMsg, DVIDException

# Must exactly match the fields from dvid: /api/server/info
# Omitting "Server uptime" because it takes a lot of space.
SERVER_INFO_FIELDS =  ["DVID Version", "Datastore Version", "Cores", "Maximum Cores", "Storage backend"]
TREEVIEW_COLUMNS = ["Alias", "UUID", "TypeName", "Details"]

class ContentsBrowser(QDialog):
    """
    Displays the contents of a DVID server, listing all datasets and the volumes/nodes within each dataset.
    The user's selected dataset, volume, and node can be accessed via the `get_selection()` method.
    
    If the dialog is constructed with mode='specify_new', then the user is asked to provide a new data name, 
    and choose the dataset and node to which it will belong. 
    
    **TODO:**

    * Show more details in dataset list (e.g. shape, axes, pixel type)
    * Show more details in node list (e.g. date modified, parents, children)
    * Gray-out nodes that aren't "open" for adding new volumes
    """
    def __init__(self, suggested_hostnames, mode='select_existing', parent=None):
        """
        Constructor.
        
        suggested_hostnames: A list of hostnames to suggest to the user, e.g. ["localhost:8000"]
        mode: Either 'select_existing' or 'specify_new'
        parent: The parent widget.
        """
        super( ContentsBrowser, self ).__init__(parent)
        self._suggested_hostnames = suggested_hostnames
        self._mode = mode
        self._current_dset = None
        self._repos_info = None
        self._hostname = None
        
        # Create the UI
        self._init_layout()

    VolumeSelection = collections.namedtuple( "VolumeSelection", "hostname dataset_uuid data_name node_uuid typename" )
    def get_selection(self):
        """
        Get the user's current (or final) selection.
        Returns a VolumeSelection tuple.
        """
        node_uuid = self._get_selected_node()
        dset_uuid, data_name, typename = self._get_selected_data()
        
        if self._mode == "specify_new":
            data_name = str( self._new_data_edit.text() )
        
        return ContentsBrowser.VolumeSelection(self._hostname, dset_uuid, data_name, node_uuid, typename)

    def _init_layout(self):
        """
        Create the GUI widgets (but leave them empty).
        """
        hostname_combobox = QComboBox(parent=self)
        self._hostname_combobox = hostname_combobox
        hostname_combobox.setEditable(True)
        hostname_combobox.setSizePolicy( QSizePolicy.Expanding, QSizePolicy.Maximum )
        hostname_combobox.installEventFilter(self)
        for hostname in self._suggested_hostnames:
            hostname_combobox.addItem( hostname )

        self._connect_button = QPushButton("Connect", parent=self, clicked=self._handle_new_hostname)

        hostname_layout = QHBoxLayout()
        hostname_layout.addWidget( hostname_combobox )
        hostname_layout.addWidget( self._connect_button )

        hostinfo_table = QTableWidget()
        hostinfo_table.setColumnCount(len(SERVER_INFO_FIELDS))
        hostinfo_table.setHorizontalHeaderLabels(SERVER_INFO_FIELDS)
        hostinfo_table.horizontalHeader().setVisible(True)
        hostinfo_table.verticalHeader().setVisible(False)
        hostinfo_table.setRowCount(1)
        hostinfo_table.setItem(0,0, QTableWidgetItem("Placeholder"))
        hostinfo_table.setVisible(False)
        hostinfo_table.resizeRowsToContents()
        hostinfo_table.horizontalHeader().setStretchLastSection(True)
        table_height = hostinfo_table.verticalHeader().sectionSize(0) + hostinfo_table.rowHeight(0)
        hostinfo_table.resize( QSize( hostinfo_table.width(), table_height ) )
        hostinfo_table.setMaximumSize( QSize( 1000, table_height ) )
        hostinfo_table.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        host_layout = QVBoxLayout()
        host_layout.addLayout(hostname_layout)
        host_layout.addWidget(hostinfo_table)

        host_groupbox = QGroupBox("DVID Host", parent=self)
        host_groupbox.setLayout( host_layout )
        host_groupbox.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Preferred )
        
        data_treewidget = QTreeWidget(parent=self)
        data_treewidget.setHeaderLabels( TREEVIEW_COLUMNS ) # TODO: Add type, shape, axes, etc.
        data_treewidget.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Preferred )
        data_treewidget.itemSelectionChanged.connect( self._handle_data_selection )

        data_layout = QVBoxLayout()
        data_layout.addWidget( data_treewidget )
        data_groupbox = QGroupBox("Data Volumes", parent=self)
        data_groupbox.setLayout( data_layout )
        
        node_listwidget = QListWidget(parent=self)
        node_listwidget.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Preferred )
        node_listwidget.itemSelectionChanged.connect( self._update_display )

        node_layout = QVBoxLayout()
        node_layout.addWidget( node_listwidget )
        node_groupbox = QGroupBox("Nodes", parent=self)
        node_groupbox.setLayout( node_layout )

        new_data_edit = QLineEdit(parent=self)
        new_data_edit.textEdited.connect( self._update_display )
        full_url_label = QLabel(parent=self)
        full_url_label.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Maximum )
        text_flags = full_url_label.textInteractionFlags()
        full_url_label.setTextInteractionFlags( text_flags | Qt.TextSelectableByMouse )

        new_data_layout = QVBoxLayout()
        new_data_layout.addWidget( new_data_edit )
        new_data_groupbox = QGroupBox("New Data Volume", parent=self)
        new_data_groupbox.setLayout( new_data_layout )
        new_data_groupbox.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Maximum )

        buttonbox = QDialogButtonBox( Qt.Horizontal, parent=self )
        buttonbox.setStandardButtons( QDialogButtonBox.Ok | QDialogButtonBox.Cancel )
        buttonbox.accepted.connect( self.accept )
        buttonbox.rejected.connect( self.reject )
        buttonbox.button(QDialogButtonBox.Ok).setEnabled(False)

        layout = QVBoxLayout()
        layout.addWidget( host_groupbox )
        layout.addWidget( data_groupbox )
        layout.addWidget( node_groupbox )
        if self._mode == "specify_new":
            layout.addWidget( new_data_groupbox )
        else:
            new_data_groupbox.hide()
        layout.addWidget( full_url_label )
        layout.addWidget( buttonbox )

        # Stretch factors
        layout.setStretchFactor(data_groupbox, 3)
        layout.setStretchFactor(node_groupbox, 1)
        
        self.setLayout(layout)
        self.setWindowTitle( "Select DVID Volume" )

        # Initially disabled
        data_groupbox.setEnabled(False)
        node_groupbox.setEnabled(False)
        new_data_groupbox.setEnabled(False)

        # Save instance members
        self._hostinfo_table = hostinfo_table
        self._data_groupbox = data_groupbox
        self._node_groupbox = node_groupbox
        self._new_data_groupbox = new_data_groupbox
        self._data_treewidget = data_treewidget
        self._node_listwidget = node_listwidget
        self._new_data_edit = new_data_edit
        self._full_url_label = full_url_label
        self._buttonbox = buttonbox

    def sizeHint(self):
        return QSize(700, 500)
    
    def eventFilter(self, watched, event):
        if watched == self._hostname_combobox \
        and event.type() == QEvent.KeyPress \
        and ( event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter):
            self._connect_button.click()
            return True
        return False

    def showEvent(self, event):
        """
        Raise the window when it is shown.
        For some reason, that doesn't happen automatically if this widget is also the main window.
        """
        super(ContentsBrowser, self).showEvent(event)
        self.raise_()

    def _handle_new_hostname(self):
        new_hostname = str( self._hostname_combobox.currentText() )
        if '://' in new_hostname:
            new_hostname = new_hostname.split('://')[1] 

        error_msg = None
        self._server_info = None
        self._repos_info = None
        self._current_dset = None
        self._hostname = None

        try:
            # Query the server
            connection = DVIDConnection(new_hostname)
            self._server_info = ContentsBrowser._get_server_info(connection)
            self._repos_info = ContentsBrowser._get_repos_info(connection)
            self._hostname = new_hostname
            self._connection = connection
        except DVIDException as ex:
            error_msg = "libdvid.DVIDException: {}".format( ex.message )
        except ErrMsg as ex:
            error_msg = "libdvid.ErrMsg: {}".format( ex.message )

        if error_msg:
            QMessageBox.critical(self, "Connection Error", error_msg)
            self._populate_node_list(-1)
        else:
            self._connect_button.setEnabled(False)
            self._buttonbox.button(QDialogButtonBox.Ok).setEnabled(True)

        enable_contents = self._repos_info is not None
        self._data_groupbox.setEnabled(enable_contents)
        self._node_groupbox.setEnabled(enable_contents)
        self._new_data_groupbox.setEnabled(enable_contents)

        self._populate_hostinfo_table()
        self._populate_datasets_tree()

    @classmethod
    def _get_server_info(self, connection):
        status, body, error_message = connection.make_request( "/server/info", ConnectionMethod.GET);
        assert status == httplib.OK, "Request for /server/info returned status {}".format( status )
        assert error_message == ""
        server_info = json.loads(body)
        return server_info

    @classmethod
    def _get_repos_info(self, connection):
        status, body, error_message = connection.make_request( "/repos/info", ConnectionMethod.GET)
        assert status == httplib.OK, "Request for /repos/info returned status {}".format( status )
        assert error_message == ""
        repos_info = json.loads(body)
        
        # Discard uuids with 'null' content (I don't know why they sometimes exist...)
        repos_info = filter( lambda (uuid, repo_info): repo_info, repos_info.items() )
        return collections.OrderedDict(sorted(repos_info))
    
    def _populate_hostinfo_table(self):
        self._hostinfo_table.setVisible( self._server_info is not None )

        for column_index, fieldname in enumerate(SERVER_INFO_FIELDS):
            try:
                field = self._server_info[fieldname]
            except KeyError:
                field = "<UNDEFINED>"
            item = QTableWidgetItem(field)
            flags = item.flags()
            flags &= ~Qt.ItemIsSelectable
            flags &= ~Qt.ItemIsEditable
            item.setFlags( flags )
            self._hostinfo_table.setItem(0, column_index, item)
        self._hostinfo_table.resizeColumnsToContents()
        self._hostinfo_table.horizontalHeader().setStretchLastSection(True) # Force refresh of last column.

    def _populate_datasets_tree(self):
        """
        Initialize the tree widget of datasets and volumes.
        """
        self._data_treewidget.clear()
        
        if self._repos_info is None:
            return
        
        for repo_uuid, repo_info in sorted(self._repos_info.items()):
            if repo_info is None:
                continue
            repo_column_dict = collections.defaultdict(str)
            repo_column_dict["Alias"] = repo_info["Alias"]
            repo_column_dict["Details"] = "Created: " + repo_info["Created"]
            repo_column_dict["UUID"] = repo_uuid
            repo_column_values = [repo_column_dict[k] for k in TREEVIEW_COLUMNS]
            dset_item = QTreeWidgetItem( self._data_treewidget, QStringList( repo_column_values ) )
            dset_item.setData( 0, Qt.UserRole, (repo_uuid, "", "") )
            for data_name, data_info in repo_info["DataInstances"].items():
                data_instance_dict = collections.defaultdict(str)
                data_instance_dict["Alias"] = data_name
                typename = data_info["Base"]["TypeName"]
                data_instance_dict["TypeName"] = typename

                is_roi = (typename == 'roi')
                is_voxels = (typename in ['labelblk', 'uint8blk'])
                if is_voxels:
                    start_coord = data_info["Extended"]["MinPoint"]
                    if start_coord:
                        start_coord = tuple(start_coord)
                    stop_coord = data_info["Extended"]["MaxPoint"]
                    if stop_coord:
                        stop_coord = tuple(x+1 for x in stop_coord)
                    if start_coord and stop_coord:
                        shape = tuple(b - a for a,b in zip(start_coord, stop_coord))
                    else:
                        shape = None
                    data_instance_dict["Details"] = "Size={} | Start={} | Stop={}".format( shape, start_coord, stop_coord )

                data_column_values = [data_instance_dict[k] for k in TREEVIEW_COLUMNS]
                data_item = QTreeWidgetItem( dset_item, data_column_values )
                data_item.setData( 0, Qt.UserRole, (repo_uuid, data_name, typename) )

                # If we're in specify_new mode, only the
                # dataset parent items are selectable.
                # Also, non-volume items aren't selectable.
                if self._mode == 'specify_new' or not (is_voxels or is_roi):
                    flags = data_item.flags()
                    flags &= ~Qt.ItemIsSelectable
                    flags &= ~Qt.ItemIsEnabled
                    data_item.setFlags( flags )
        
        self._data_treewidget.collapseAll()
        self._data_treewidget.setSortingEnabled(True)
        
        # Select the first item by default.
        if self._mode == "select_existing":
            first_item = self._data_treewidget.topLevelItem(0).child(0)
        else:
            first_item = self._data_treewidget.topLevelItem(0)            
        self._data_treewidget.setCurrentItem( first_item, 0 )

    def _handle_data_selection(self):
        """
        When the user clicks a new data item, respond by updating the node list.
        """
        selected_items = self._data_treewidget.selectedItems()
        if not selected_items:
            return None
        item = selected_items[0]
        item_data = item.data(0, Qt.UserRole).toPyObject()
        if not item_data:
            return
        dset_uuid, data_name, typename = item_data
        if self._current_dset != dset_uuid:
            self._populate_node_list(dset_uuid)
        
        self._update_display()

    def _populate_node_list(self, dataset_uuid):
        """
        Replace the contents of the node list widget 
        to show all the nodes for the currently selected dataset.
        """
        self._node_listwidget.clear()
        
        if self._repos_info is None:
            return
        
        # For now, we simply show the nodes in sorted order, without respect to dag order
        all_uuids = sorted( self._repos_info[dataset_uuid]["DAG"]["Nodes"].keys() )
        for node_uuid in all_uuids:
            node_item = QListWidgetItem( node_uuid, parent=self._node_listwidget )
            node_item.setData( Qt.UserRole, node_uuid )
        self._current_dset = dataset_uuid

        # Select the last one by default.
        last_row = self._node_listwidget.count() - 1
        last_item = self._node_listwidget.item( last_row )
        self._node_listwidget.setCurrentItem( last_item )
        self._update_display()

    def _get_selected_node(self):
        selected_items = self._node_listwidget.selectedItems()
        if not selected_items:
            return None
        selected_node_item = selected_items[0]
        node_item_data = selected_node_item.data(Qt.UserRole)
        return str( node_item_data.toString() )
        
    def _get_selected_data(self):
        selected_items = self._data_treewidget.selectedItems()
        if not selected_items:
            return None, None
        selected_data_item = selected_items[0]
        data_item_data = selected_data_item.data(0, Qt.UserRole).toPyObject()
        if selected_data_item:
            dset_uuid, data_name, typename = data_item_data
        else:
            dset_uuid = data_name = typename = None
        return dset_uuid, data_name, typename
    
    def _update_display(self):
        """
        Update the path label to reflect the user's currently selected uuid and new volume name.
        """
        hostname, dset_uuid, dataname, node_uuid, typename = self.get_selection()
        full_path = "http://{hostname}/api/node/{uuid}/{dataname}"\
                    "".format( hostname=self._hostname, uuid=node_uuid, dataname=dataname )
        self._full_url_label.setText( full_path )
        
        ok_button = self._buttonbox.button( QDialogButtonBox.Ok )
        ok_button.setEnabled( dataname != "" )

if __name__ == "__main__":
    """
    This main section permits simple command-line control.
    usage: contents_browser.py [-h] [--mock-server-hdf5=MOCK_SERVER_HDF5] hostname:port
    
    If --mock-server-hdf5 is provided, the mock server will be launched with the provided hdf5 file.
    Otherwise, the DVID server should already be running on the provided hostname.
    """
    import sys
    import argparse
    
    # Make the program quit on Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from PyQt4.QtGui import QApplication

    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-server-hdf5", required=False)
    parser.add_argument("--mode", choices=["select_existing", "specify_new"], default="select_existing")
    parser.add_argument("hostname", metavar="hostname:port", default="localhost:8000", nargs="?")

    DEBUG = False
    if DEBUG and len(sys.argv) == 1:
        # default debug args
        parser.print_help()
        print ""
        print "*******************************************************"
        print "No args provided.  Starting with special debug args...."
        print "*******************************************************"
        sys.argv.append("--mock-server-hdf5=/magnetic/mockdvid_gigacube.h5")
        #sys.argv.append("--mode=specify_new")
        sys.argv.append("localhost:8000")

    parsed_args = parser.parse_args()
    
    server_proc = None
    if parsed_args.mock_server_hdf5:
        from mockserver.h5mockserver import H5MockServer
        hostname, port = parsed_args.hostname.split(":")
        server_proc, shutdown_event = H5MockServer.create_and_start( parsed_args.mock_server_hdf5,
                                                     hostname,
                                                     int(port),
                                                     same_process=False,
                                                     disable_server_logging=False )
    
    app = QApplication([])
    browser = ContentsBrowser([parsed_args.hostname], parsed_args.mode)

    try:
        if browser.exec_() == QDialog.Accepted:
            print "The dialog was accepted with result: ", browser.get_selection()
        else:
            print "The dialog was rejected."
    finally:
        if server_proc:
            shutdown_event.set()
            server_proc.join()
