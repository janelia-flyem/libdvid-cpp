import os
import httplib
import json

from libdvid import DVIDConnection, ConnectionMethod
TEST_DVID_SERVER = os.getenv("TEST_DVID_SERVER", "127.0.0.1:8000")

def get_testrepo_root_uuid():
    connection = DVIDConnection(TEST_DVID_SERVER)
    status, body, error_message = connection.make_request( "/repos/info", ConnectionMethod.GET)
    assert status == httplib.OK, "Request for /repos/info returned status {}".format( status )
    assert error_message == ""
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
    connection = DVIDConnection(TEST_DVID_SERVER)
    repo_info_uri = "/repo/{uuid}/info".format( uuid=uuid )
    status, body, error_message = connection.make_request( repo_info_uri, ConnectionMethod.GET)
    assert status == httplib.OK, "Request for {} returned status {}".format(repo_info_uri, status)
    assert error_message == ""
    repo_info = json.loads(body)
    for instance_name in repo_info["DataInstances"].keys():
        status, body, error_message = connection.make_request( "/api/repo/{uuid}/{dataname}?imsure=true"
                                                               .format( uuid=uuid, dataname=str(instance_name) ),
                                                               ConnectionMethod.DELETE )            
