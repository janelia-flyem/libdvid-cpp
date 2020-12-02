#include <cstring>

#include "DVIDConnection.h"
#include "DVIDException.h"
#include <sstream>
#include <json/json.h>

extern "C" {
#include <curl/curl.h>
#include <zmq.h>
}

using std::string;
using std::stringstream;
using std::ostringstream;

/*!
 * Initializes libcurl libraries.  This could probably be a singleton
 * but the static scope make unauthorized access unlikely
*/
struct CurlEnv {
    CurlEnv() { curl_global_init(CURL_GLOBAL_ALL); }
    ~CurlEnv() { curl_global_cleanup(); }
};
static CurlEnv curl_env;

//! Function for libcurl that writes results into a string buffer
static size_t
WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
    size_t realsize = size * nmemb;
    string* str = (string*) userp;
    str->append((const char*) contents, realsize);
    return realsize;
}

static const std::map<CURLcode, std::string> CurlErrorNames =
{
    { CURLE_OK, "CURLE_OK" },
    { CURLE_UNSUPPORTED_PROTOCOL, "CURLE_UNSUPPORTED_PROTOCOL" },
    { CURLE_FAILED_INIT, "CURLE_FAILED_INIT" },
    { CURLE_URL_MALFORMAT, "CURLE_URL_MALFORMAT" },
    { CURLE_NOT_BUILT_IN, "CURLE_NOT_BUILT_IN" },
    { CURLE_COULDNT_RESOLVE_PROXY, "CURLE_COULDNT_RESOLVE_PROXY" },
    { CURLE_COULDNT_RESOLVE_HOST, "CURLE_COULDNT_RESOLVE_HOST" },
    { CURLE_COULDNT_CONNECT, "CURLE_COULDNT_CONNECT" },
    //{ CURLE_WEIRD_SERVER_REPLY, "CURLE_WEIRD_SERVER_REPLY" },
    { CURLE_REMOTE_ACCESS_DENIED, "CURLE_REMOTE_ACCESS_DENIED" },
    { CURLE_FTP_ACCEPT_FAILED, "CURLE_FTP_ACCEPT_FAILED" },
    { CURLE_FTP_WEIRD_PASS_REPLY, "CURLE_FTP_WEIRD_PASS_REPLY" },
    { CURLE_FTP_ACCEPT_TIMEOUT, "CURLE_FTP_ACCEPT_TIMEOUT" },
    { CURLE_FTP_WEIRD_PASV_REPLY, "CURLE_FTP_WEIRD_PASV_REPLY" },
    { CURLE_FTP_WEIRD_227_FORMAT, "CURLE_FTP_WEIRD_227_FORMAT" },
    { CURLE_FTP_CANT_GET_HOST, "CURLE_FTP_CANT_GET_HOST" },
    { CURLE_HTTP2, "CURLE_HTTP2" },
    { CURLE_FTP_COULDNT_SET_TYPE, "CURLE_FTP_COULDNT_SET_TYPE" },
    { CURLE_PARTIAL_FILE, "CURLE_PARTIAL_FILE" },
    { CURLE_FTP_COULDNT_RETR_FILE, "CURLE_FTP_COULDNT_RETR_FILE" },
    { CURLE_OBSOLETE20, "CURLE_OBSOLETE20" },
    { CURLE_QUOTE_ERROR, "CURLE_QUOTE_ERROR" },
    { CURLE_HTTP_RETURNED_ERROR, "CURLE_HTTP_RETURNED_ERROR" },
    { CURLE_WRITE_ERROR, "CURLE_WRITE_ERROR" },
    { CURLE_OBSOLETE24, "CURLE_OBSOLETE24" },
    { CURLE_UPLOAD_FAILED, "CURLE_UPLOAD_FAILED" },
    { CURLE_READ_ERROR, "CURLE_READ_ERROR" },
    { CURLE_OUT_OF_MEMORY, "CURLE_OUT_OF_MEMORY" },
    { CURLE_OPERATION_TIMEDOUT, "CURLE_OPERATION_TIMEDOUT" },
    { CURLE_OBSOLETE29, "CURLE_OBSOLETE29" },
    { CURLE_FTP_PORT_FAILED, "CURLE_FTP_PORT_FAILED" },
    { CURLE_FTP_COULDNT_USE_REST, "CURLE_FTP_COULDNT_USE_REST" },
    { CURLE_OBSOLETE32, "CURLE_OBSOLETE32" },
    { CURLE_RANGE_ERROR, "CURLE_RANGE_ERROR" },
    { CURLE_HTTP_POST_ERROR, "CURLE_HTTP_POST_ERROR" },
    { CURLE_SSL_CONNECT_ERROR, "CURLE_SSL_CONNECT_ERROR" },
    { CURLE_BAD_DOWNLOAD_RESUME, "CURLE_BAD_DOWNLOAD_RESUME" },
    { CURLE_FILE_COULDNT_READ_FILE, "CURLE_FILE_COULDNT_READ_FILE" },
    { CURLE_LDAP_CANNOT_BIND, "CURLE_LDAP_CANNOT_BIND" },
    { CURLE_LDAP_SEARCH_FAILED, "CURLE_LDAP_SEARCH_FAILED" },
    { CURLE_OBSOLETE40, "CURLE_OBSOLETE40" },
    { CURLE_FUNCTION_NOT_FOUND, "CURLE_FUNCTION_NOT_FOUND" },
    { CURLE_ABORTED_BY_CALLBACK, "CURLE_ABORTED_BY_CALLBACK" },
    { CURLE_BAD_FUNCTION_ARGUMENT, "CURLE_BAD_FUNCTION_ARGUMENT" },
    { CURLE_OBSOLETE44, "CURLE_OBSOLETE44" },
    { CURLE_INTERFACE_FAILED, "CURLE_INTERFACE_FAILED" },
    { CURLE_OBSOLETE46, "CURLE_OBSOLETE46" },
    { CURLE_TOO_MANY_REDIRECTS, "CURLE_TOO_MANY_REDIRECTS" },
    { CURLE_UNKNOWN_OPTION, "CURLE_UNKNOWN_OPTION" },
    { CURLE_TELNET_OPTION_SYNTAX, "CURLE_TELNET_OPTION_SYNTAX" },
    { CURLE_OBSOLETE50, "CURLE_OBSOLETE50" },
    { CURLE_PEER_FAILED_VERIFICATION, "CURLE_PEER_FAILED_VERIFICATION" },
    { CURLE_GOT_NOTHING, "CURLE_GOT_NOTHING" },
    { CURLE_SSL_ENGINE_NOTFOUND, "CURLE_SSL_ENGINE_NOTFOUND" },
    { CURLE_SSL_ENGINE_SETFAILED, "CURLE_SSL_ENGINE_SETFAILED" },
    { CURLE_SEND_ERROR, "CURLE_SEND_ERROR" },
    { CURLE_RECV_ERROR, "CURLE_RECV_ERROR" },
    { CURLE_OBSOLETE57, "CURLE_OBSOLETE57" },
    { CURLE_SSL_CERTPROBLEM, "CURLE_SSL_CERTPROBLEM" },
    { CURLE_SSL_CIPHER, "CURLE_SSL_CIPHER" },
    { CURLE_SSL_CACERT, "CURLE_SSL_CACERT" },
    { CURLE_BAD_CONTENT_ENCODING, "CURLE_BAD_CONTENT_ENCODING" },
    { CURLE_LDAP_INVALID_URL, "CURLE_LDAP_INVALID_URL" },
    { CURLE_FILESIZE_EXCEEDED, "CURLE_FILESIZE_EXCEEDED" },
    { CURLE_USE_SSL_FAILED, "CURLE_USE_SSL_FAILED" },
    { CURLE_SEND_FAIL_REWIND, "CURLE_SEND_FAIL_REWIND" },
    { CURLE_SSL_ENGINE_INITFAILED, "CURLE_SSL_ENGINE_INITFAILED" },
    { CURLE_LOGIN_DENIED, "CURLE_LOGIN_DENIED" },
    { CURLE_TFTP_NOTFOUND, "CURLE_TFTP_NOTFOUND" },
    { CURLE_TFTP_PERM, "CURLE_TFTP_PERM" },
    { CURLE_REMOTE_DISK_FULL, "CURLE_REMOTE_DISK_FULL" },
    { CURLE_TFTP_ILLEGAL, "CURLE_TFTP_ILLEGAL" },
    { CURLE_TFTP_UNKNOWNID, "CURLE_TFTP_UNKNOWNID" },
    { CURLE_REMOTE_FILE_EXISTS, "CURLE_REMOTE_FILE_EXISTS" },
    { CURLE_TFTP_NOSUCHUSER, "CURLE_TFTP_NOSUCHUSER" },
    { CURLE_CONV_FAILED, "CURLE_CONV_FAILED" },
    { CURLE_CONV_REQD, "CURLE_CONV_REQD" },
    { CURLE_SSL_CACERT_BADFILE, "CURLE_SSL_CACERT_BADFILE" },
    { CURLE_REMOTE_FILE_NOT_FOUND, "CURLE_REMOTE_FILE_NOT_FOUND" },
    { CURLE_SSH, "CURLE_SSH" },
    { CURLE_SSL_SHUTDOWN_FAILED, "CURLE_SSL_SHUTDOWN_FAILED" },
    { CURLE_AGAIN, "CURLE_AGAIN" },
    { CURLE_SSL_CRL_BADFILE, "CURLE_SSL_CRL_BADFILE" },
    { CURLE_SSL_ISSUER_ERROR, "CURLE_SSL_ISSUER_ERROR" },
    { CURLE_FTP_PRET_FAILED, "CURLE_FTP_PRET_FAILED" },
    { CURLE_RTSP_CSEQ_ERROR, "CURLE_RTSP_CSEQ_ERROR" },
    { CURLE_RTSP_SESSION_ERROR, "CURLE_RTSP_SESSION_ERROR" },
    { CURLE_FTP_BAD_FILE_LIST, "CURLE_FTP_BAD_FILE_LIST" },
    { CURLE_CHUNK_FAILED, "CURLE_CHUNK_FAILED" },
    { CURLE_NO_CONNECTION_AVAILABLE, "CURLE_NO_CONNECTION_AVAILABLE" },
    { CURLE_SSL_PINNEDPUBKEYNOTMATCH, "CURLE_SSL_PINNEDPUBKEYNOTMATCH" },
    { CURLE_SSL_INVALIDCERTSTATUS, "CURLE_SSL_INVALIDCERTSTATUS" },
    //{ CURLE_HTTP2_STREAM, "CURLE_HTTP2_STREAM" }
};

namespace libdvid {

const int DVIDConnection::DEFAULT_TIMEOUT;

//! Defines DVID prefix -- this might have a version ID eventually 
const char* DVIDConnection::DVID_PREFIX = "/api";

DVIDConnection::DVIDConnection(string addr_, string user, string app,
        string resource_server_, int resource_port_) : 
    addr(addr_), username(user), appname(app), resource_server(resource_server_),
    resource_port(resource_port_)
{
    // trim trailing slash, e.g. http://localhost:8000/
    while (*addr.rbegin() == '/')
    {
        addr = addr.substr(0, addr.size()-1);
    }

    curl_connection = curl_easy_init();
    
    if (resource_server != "") {
        setup_0mq();
    }
}

void DVIDConnection::setup_0mq()
{
    zmq_context = zmq_ctx_new();
    zmq_commsocket = zmq_socket(zmq_context, ZMQ_REQ);
    ostringstream addr0mq;
    addr0mq << "tcp://" << resource_server << ":" << resource_port;
    zmq_connect(zmq_commsocket, addr0mq.str().c_str()); 
}

DVIDConnection::DVIDConnection(const DVIDConnection& copy_connection)
{
    addr = copy_connection.addr;
    username = copy_connection.username;
    appname = copy_connection.appname;
    resource_server = copy_connection.resource_server;
    resource_port = copy_connection.resource_port;
    curl_connection = curl_easy_init();
    if (resource_server != "") {
        setup_0mq();
    }
}

DVIDConnection::~DVIDConnection()
{
    curl_easy_cleanup(curl_connection);

    if (resource_server != "") {
        zmq_close(zmq_commsocket);
        zmq_ctx_destroy(zmq_context);
    }
}

int DVIDConnection::make_head_request(string endpoint) {
    std::string error_msg;
    return make_request(endpoint, HEAD, BinaryDataPtr(), BinaryData::create_binary_data(), error_msg);
}

int DVIDConnection::make_request(string endpoint, ConnectionMethod method,
        BinaryDataPtr payload, BinaryDataPtr results, string& error_msg,
        ConnectionType type, int timeout, unsigned long long datasize, bool checkHttpErrors)
{
    CURLcode result;
    char buffer[100];

    // pass the custom headers
    struct curl_slist *headers=0;
    if (type == JSON) {
        headers = curl_slist_append(headers, "Content-Type: application/json");
    } else if (type == BINARY) {
        headers = curl_slist_append(headers, "Content-Type: application/octet-stream");
    } 
    curl_easy_setopt(curl_connection, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl_connection, CURLOPT_NOSIGNAL, 1);

    // handle any special characters
    char * user_enc = curl_easy_escape(curl_connection, username.c_str(), username.length());
    char * app_enc = curl_easy_escape(curl_connection, appname.c_str(), appname.length());

    // append user and app info for dvid requests
    if (endpoint.find('?') == string::npos) {
        // create query string
        endpoint += "?u=" + string(user_enc) + "&app=" + string(app_enc);
    } else {
        // add to query string
        endpoint += "&u=" + string(user_enc) + "&app=" + string(app_enc);
    }

    // load url
    string url = get_uri_root() + endpoint;
    curl_easy_setopt(curl_connection, CURLOPT_URL, url.c_str());
        
    // set the method
    if (method == HEAD) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "HEAD");
    }
    else if (method == GET) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "GET");
    }
    else if (method == POST) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "POST");
    }
    else if (method == PUT) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "PUT");
    }
    else if (method == DELETE) {
        curl_easy_setopt(curl_connection, CURLOPT_CUSTOMREQUEST, "DELETE");
    }
    else {
        throw std::runtime_error("Bad connection method");
    }


    // HEAD request has no body.
    curl_easy_setopt(curl_connection, CURLOPT_NOBODY, long(method == HEAD));

    // set to 0 for infinite
    curl_easy_setopt(curl_connection, CURLOPT_TIMEOUT, long(timeout));

    // post binary data
    if (payload) {
        // set binary payload and indicate size
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDS, payload->get_raw());
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDSIZE, long(payload->length()));
    } else {
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDS, 0);
        curl_easy_setopt(curl_connection, CURLOPT_POSTFIELDSIZE, long(0));
    }
    
    // results should be an empty binary array
    assert(results);
    assert(results->length() == 0);

    // pass the raw pointer to the write data command
    string& raw_data = results->get_data();

    // set callback for writing data
    curl_easy_setopt(curl_connection, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
    curl_easy_setopt(curl_connection, CURLOPT_WRITEDATA, (void *)&raw_data);

    // set verbose only for debug
    //curl_easy_setopt(curl_connection, CURLOPT_VERBOSE, 1L);

    // set error buffer
    char error_buf[CURL_ERROR_SIZE]; // size of the error buffer
    memset(error_buf, 0, CURL_ERROR_SIZE);
    curl_easy_setopt(curl_connection, CURLOPT_ERRORBUFFER, error_buf);

    // request resource if resource server is available 
    int client_id; 
    if (resource_server != "") {
        Json::Value request;
        request["type"] = "request"; 
        request["resource"] = addr;
        request["numopts"] = 1;
        if ((method == POST) || (method == PUT)) {
            unsigned long long payload_size = 0;
            request["read"] = false;
            if (payload) {
                payload_size = (unsigned long long)(payload->length());
            }
            request["datasize"] = static_cast<Json::UInt64>(payload_size);
        } else {
            request["read"] = true;
            request["datasize"] = static_cast<Json::UInt64>(datasize); // need to pass if read unless really tiny
        }
        ostringstream ss;
        ss << request;
        string request_str = ss.str();
   
        // send request
        zmq_send(zmq_commsocket, request_str.c_str(), request_str.length(), 0);
    
        // retrieve request
        zmq_recv(zmq_commsocket, buffer, 100, 0);

        // parse response
        Json::Value response;
        stringstream stream_buffer;
        stream_buffer << string(buffer);
        stream_buffer >> response;
        client_id = response["id"].asInt();
        bool resp_val = response["available"].asBool();
        if (!resp_val) {
            // if not availabe, create SUB, filter on port, wait for message
            void* zmq_sub = zmq_socket(zmq_context, ZMQ_SUB);
            ostringstream addr0mq;
            addr0mq << "tcp://" << resource_server << ":" << resource_port + 1;
            ostringstream cidstream;
            cidstream << client_id;
            zmq_setsockopt(zmq_sub, ZMQ_SUBSCRIBE, cidstream.str().c_str(), cidstream.str().length());
            zmq_connect(zmq_sub, addr0mq.str().c_str());

            // listen for events
            zmq_recv(zmq_sub, buffer, 100, 0);
            zmq_close(zmq_sub);

            // when message is received, send reservation requst
            ostringstream holdstream;
            holdstream << "{\"type\": \"hold\", \"id\": " << client_id << '}';
            string holdstr = holdstream.str();
            zmq_send(zmq_commsocket, holdstr.c_str(), holdstr.length(), 0);
            zmq_recv(zmq_commsocket, buffer, 100, 0);
        }
    } 
    
    
    
    // actually perform the request
    result = curl_easy_perform(curl_connection);

    // release resource if resource server is available 
    if (resource_server != "") {
        ostringstream cidstream;
        cidstream << "{\"type\": \"release\", \"id\": " << client_id << '}';
        string cidstr = cidstream.str();
        zmq_send(zmq_commsocket, cidstr.c_str(), cidstr.length(), 0);
        zmq_recv(zmq_commsocket, buffer, 100, 0);
    } 
    
    // get the error code
    long http_code = 0;
    curl_easy_getinfo (curl_connection, CURLINFO_RESPONSE_CODE, &http_code);
   
    // free allocated string
    curl_free(user_enc);
    curl_free(app_enc);
    curl_slist_free_all(headers);

    // throw exception if connection doesn't work
    if (result != CURLE_OK) {
        throw DVIDException("DVIDConnection error (" + CurlErrorNames.at(result) + "): " + string(url), http_code);
    }

    // load error if there is one
    error_msg = error_buf;

    if (checkHttpErrors && int(http_code) != 200) {
        throw DVIDException("DVIDException for " + endpoint + "\n" + error_msg + "\n" + raw_data, http_code);
    }

    // return status
    return int(http_code);
}

}
