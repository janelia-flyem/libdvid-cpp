import os
import httplib
import json

from libdvid import DVIDConnection, ConnectionMethod
TEST_DVID_SERVER = os.getenv("TEST_DVID_SERVER", "127.0.0.1:8000")

def get_testrepo_root_uuid():
    connection = DVIDConnection(TEST_DVID_SERVER, "test1@blah.com", "myapp")
    status, body, _error_message = connection.make_request( "/repos/info", ConnectionMethod.GET)
    repos_info = json.loads(body)
    test_repos = filter( lambda (uuid, repo_info): repo_info and repo_info['Alias'] == 'testrepo', 
                         repos_info.items() )
    if test_repos:
        uuid = test_repos[0][0]
        return str(uuid)
    else:
        from libdvid import DVIDServerService
        server = DVIDServerService(TEST_DVID_SERVER)
        uuid = server.create_new_repo("testrepo", "This repo is for unit tests to use and abuse.");
        return str(uuid)

def delete_all_data_instances(uuid):
    connection = DVIDConnection(TEST_DVID_SERVER, "test1@blah.com", "myapp")
    repo_info_uri = "/repo/{uuid}/info".format( uuid=uuid )
    status, body, _error_message = connection.make_request( repo_info_uri, ConnectionMethod.GET)
    repo_info = json.loads(body)
    for instance_name in repo_info["DataInstances"].keys():
        status, body, _error_message = connection.make_request( "/repo/{uuid}/{dataname}?imsure=true"
                                                               .format( uuid=uuid, dataname=str(instance_name) ),
                                                               ConnectionMethod.DELETE, checkHttpErrors=False )            
