"""
This module implements a simple widget for viewing the list of repos and nodes in a DVID instance.
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
# Omitting "Server uptime" because it takes a lot of column space.
SERVER_INFO_FIELDS =  ["DVID Version", "Datastore Version", "Cores", "Maximum Cores", "Storage backend"]
TREEVIEW_COLUMNS = ["Alias", "UUID", "TypeName", "Details"]

class ContentsBrowser(QDialog):
    """
    Displays the contents of a DVID server, listing all repos and the volumes/nodes within each repo.
    The user's selected repo, volume, and node can be accessed via the `get_selection()` method.
    
    If the dialog is constructed with mode='specify_new', then the user is asked to provide a new data name, 
    and choose the repo and node to which it will belong. 
    
    **TODO:**

    * Show more details in node list (e.g. date modified, parents, children)
    * Gray-out nodes that aren't "open" for adding new volumes
    """
    def __init__(self, suggested_hostnames, default_nodes=None, mode='select_existing', parent=None):
        """
        Constructor.
        
        suggested_hostnames: A list of hostnames to suggest to the user, e.g. ["localhost:8000"]
        default_nodes: A dict of {hostname : uuid} specifying which node to auto-select for each possible host.
        mode: Either 'select_existing' or 'specify_new'
        parent: The parent widget.
        """
        super( ContentsBrowser, self ).__init__(parent)
        self._suggested_hostnames = suggested_hostnames
        self._default_nodes = default_nodes
        self._mode = mode
        self._current_repo = None
        self._repos_info = None
        self._hostname = None
        
        # Create the UI
        self._init_layout()

    VolumeSelection = collections.namedtuple( "VolumeSelection", "hostname repo_uuid data_name node_uuid typename" )
    def get_selection(self):
        """
        Get the user's current (or final) selection.
        Returns a VolumeSelection tuple.
        """
        node_uuid = self._get_selected_node()
        repo_uuid, data_name, typename = self._get_selected_data()
        
        if self._mode == "specify_new":
            data_name = str( self._new_data_edit.text() )
        
        return ContentsBrowser.VolumeSelection(self._hostname, repo_uuid, data_name, node_uuid, typename)

    def _init_layout(self):
        """
        Create the GUI widgets (but leave them empty).
        """
        hostname_combobox = QComboBox(parent=self)
        self._hostname_combobox = hostname_combobox
        hostname_combobox.setEditable(True)
        hostname_combobox.setSizePolicy( QSizePolicy.Expanding, QSizePolicy.Maximum )
        for hostname in self._suggested_hostnames:
            hostname_combobox.addItem( hostname )

        # EventFilter is installed after everything else is initialized. (See below.)
        #hostname_combobox.installEventFilter(self)

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
        
        repo_treewidget = QTreeWidget(parent=self)
        repo_treewidget.setHeaderLabels( TREEVIEW_COLUMNS ) # TODO: Add type, shape, axes, etc.
        repo_treewidget.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Preferred )
        repo_treewidget.itemSelectionChanged.connect( self._handle_data_selection )

        data_layout = QVBoxLayout()
        data_layout.addWidget( repo_treewidget )
        data_groupbox = QGroupBox("Data Volumes", parent=self)
        data_groupbox.setLayout( data_layout )
        
        node_listwidget = QListWidget(parent=self)
        node_listwidget.setSizePolicy( QSizePolicy.Preferred, QSizePolicy.Preferred )
        node_listwidget.itemSelectionChanged.connect( self._update_status )

        node_layout = QVBoxLayout()
        node_layout.addWidget( node_listwidget )
        node_groupbox = QGroupBox("Nodes", parent=self)
        node_groupbox.setLayout( node_layout )

        new_data_edit = QLineEdit(parent=self)
        new_data_edit.textEdited.connect( self._update_status )
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
        self.resize(1000, 1000)

        # Initially disabled
        data_groupbox.setEnabled(False)
        node_groupbox.setEnabled(False)
        new_data_groupbox.setEnabled(False)

        # Save instance members
        self._hostinfo_table = hostinfo_table
        self._data_groupbox = data_groupbox
        self._node_groupbox = node_groupbox
        self._new_data_groupbox = new_data_groupbox
        self._repo_treewidget = repo_treewidget
        self._node_listwidget = node_listwidget
        self._new_data_edit = new_data_edit
        self._full_url_label = full_url_label
        self._buttonbox = buttonbox

        # Finally install eventfilter (after everything is initialized)
        hostname_combobox.installEventFilter(self)

    def sizeHint(self):
        return QSize(1000,1000)
    
    def eventFilter(self, watched, event):
        """
        When the user presses the 'Enter' key, auto-click 'Connect'.
        """
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
        """
        Called by 'Connect' button.
        Connect to the server, download the server info and repo info,
        and populate the GUI widgets with the data.
        """
        new_hostname = str( self._hostname_combobox.currentText() )
        if '://' in new_hostname:
            new_hostname = new_hostname.split('://')[1] 

        error_msg = None
        self._server_info = None
        self._repos_info = None
        self._current_repo = None
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
            self._populate_node_list(None)
        else:
            self._connect_button.setEnabled(False)
            self._buttonbox.button(QDialogButtonBox.Ok).setEnabled(True)

        enable_contents = self._repos_info is not None
        self._data_groupbox.setEnabled(enable_contents)
        self._node_groupbox.setEnabled(enable_contents)
        self._new_data_groupbox.setEnabled(enable_contents)

        self._populate_hostinfo_table()
        self._populate_repo_tree()
        
    @classmethod
    def _get_server_info(cls, connection):
        status, body, error_message = connection.make_request( "/server/info", ConnectionMethod.GET);
        assert status == httplib.OK, "Request for /server/info returned status {}".format( status )
        assert error_message == ""
        server_info = json.loads(body)
        return server_info

    @classmethod
    def _get_repos_info(cls, connection):
        status, body, error_message = connection.make_request( "/repos/info", ConnectionMethod.GET)
        assert status == httplib.OK, "Request for /repos/info returned status {}".format( status )
        assert error_message == ""
        repos_info = json.loads(body)
        
        # Discard uuids with 'null' content (I don't know why they sometimes exist...)
        repos_info = filter( lambda (uuid, repo_info): repo_info, repos_info.items() )
        return collections.OrderedDict(sorted(repos_info))
    
    def _populate_hostinfo_table(self):
        self._hostinfo_table.setVisible( self._server_info is not None )
        if not self._server_info:
            return

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

    def _populate_repo_tree(self):
        """
        Initialize the tree widget of repos and volumes.
        """
        self._repo_treewidget.clear()
        
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
            repo_item = QTreeWidgetItem( self._repo_treewidget, QStringList( repo_column_values ) )
            repo_item.setData( 0, Qt.UserRole, (repo_uuid, "", "") )
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
                data_item = QTreeWidgetItem( repo_item, data_column_values )
                data_item.setData( 0, Qt.UserRole, (repo_uuid, data_name, typename) )

                # If we're in specify_new mode, only the
                # repo parent items are selectable.
                # Also, non-volume items aren't selectable.
                if self._mode == 'specify_new' or not (is_voxels or is_roi):
                    flags = data_item.flags()
                    flags &= ~Qt.ItemIsSelectable
                    flags &= ~Qt.ItemIsEnabled
                    data_item.setFlags( flags )
        
        self._repo_treewidget.collapseAll()
        self._repo_treewidget.setSortingEnabled(True)

        if self._hostname in self._default_nodes:
            self._select_node_uuid(self._default_nodes[self._hostname])

        self._repo_treewidget.resizeColumnToContents(0)

    def _handle_data_selection(self):
        """
        When the user clicks a new data item, respond by updating the node list.
        """
        selected_items = self._repo_treewidget.selectedItems()
        if not selected_items:
            return None
        item = selected_items[0]
        item_data = item.data(0, Qt.UserRole).toPyObject()
        if not item_data:
            return
        repo_uuid, data_name, typename = item_data
        if self._current_repo != repo_uuid:
            self._populate_node_list(repo_uuid)
        
        self._update_status()

    def _populate_node_list(self, repo_uuid):
        """
        Replace the contents of the node list widget 
        to show all the nodes for the currently selected repo.
        """
        self._node_listwidget.clear()
        
        if self._repos_info is None or repo_uuid is None:
            return
        
        # For now, we simply show the nodes in sorted order, without respect to dag order
        all_uuids = sorted( self._repos_info[repo_uuid]["DAG"]["Nodes"].keys() )
        for node_uuid in all_uuids:
            node_item = QListWidgetItem( node_uuid, parent=self._node_listwidget )
            node_item.setData( Qt.UserRole, node_uuid )
        self._current_repo = repo_uuid

        # Select the last one by default.
        last_row = self._node_listwidget.count() - 1
        last_item = self._node_listwidget.item( last_row )
        self._node_listwidget.setCurrentItem( last_item )
        self._update_status()

    def _get_selected_node(self):
        """
        Return the currently selected node uuid.
        """
        selected_items = self._node_listwidget.selectedItems()
        if not selected_items:
            return None
        selected_node_item = selected_items[0]
        node_item_data = selected_node_item.data(Qt.UserRole)
        return str( node_item_data.toString() )
        
    def _get_selected_data(self):
        """
        Return the repo, data name, and type of the currently selected data volume (or ROI).
        """
        selected_items = self._repo_treewidget.selectedItems()
        if not selected_items:
            return None, None
        selected_data_item = selected_items[0]
        data_item_data = selected_data_item.data(0, Qt.UserRole).toPyObject()
        if selected_data_item:
            repo_uuid, data_name, typename = data_item_data
        else:
            repo_uuid = data_name = typename = None
        return repo_uuid, data_name, typename
    
    def _select_node_uuid(self, node_uuid):
        """
        Locate the repo that owns this uuid, and select it in the GUI.
        If the uuid can't be found, do nothing.
        """
        def select_repotree_item(repo_uuid):
            for row in range(self._repo_treewidget.topLevelItemCount()):
                repo_item = self._repo_treewidget.topLevelItem(row)
                if repo_uuid == repo_item.data(0, Qt.UserRole).toPyObject()[0]:
                    self._repo_treewidget.setCurrentItem(repo_item)
                    repo_item.setExpanded(True)
                    self._repo_treewidget.scrollTo( self._repo_treewidget.selectedIndexes()[0],
                                                    QTreeWidget.PositionAtCenter )
                    break

        def select_nodelist_item(node_uuid):
            for row in range(self._node_listwidget.count()):
                item = self._node_listwidget.item(row)
                if node_uuid == item.data(Qt.UserRole).toPyObject():
                    self._node_listwidget.setCurrentItem( item )
                    break
        
        for repo_uuid, repo_info in sorted(self._repos_info.items()):
            if node_uuid in repo_info["DAG"]["Nodes"].keys():
                # Select the right repo parent item
                select_repotree_item(repo_uuid)

                # Select the right row in the node list
                # (The node list was automatically updated when the repo selection changed, above.)
                select_nodelist_item(node_uuid)
                break

    def _update_status(self):
        """
        Update the path label to reflect the user's currently selected uuid and new volume name.
        """
        hostname, repo_uuid, dataname, node_uuid, typename = self.get_selection()
        full_path = "http://{hostname}/api/node/{uuid}/{dataname}"\
                    "".format( hostname=self._hostname, uuid=node_uuid, dataname=dataname )
        self._full_url_label.setText( full_path )
        
        ok_button = self._buttonbox.button( QDialogButtonBox.Ok )
        ok_button.setEnabled( dataname != "" )

if __name__ == "__main__":
    # Make the program quit on Ctrl+C
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    from PyQt4.QtGui import QApplication    
    app = QApplication([])
    browser = ContentsBrowser(["localhost:8000", "emdata2:7000"],
                              default_nodes={ "localhost:8000" : '57c4c6a0740d4509a02da6b9453204cb'},
                              mode="select_existing")

    if browser.exec_() == QDialog.Accepted:
        print "The dialog was accepted with result: ", browser.get_selection()
    else:
        print "The dialog was rejected."
