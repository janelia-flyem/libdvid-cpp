#include "DVIDGraph.h"

namespace libdvid {

Vertex::Vertex(Json::Value& data)
{
    id = data["Id"].asUInt();
    weight = data["Weight"].asDouble();
}

void Vertex::export_json(Json::Value& data)
{
    data["Id"] = (unsigned int)(id);
    data["Weight"] = weight;
}

Edge::Edge(Json::Value& data)
{
    id1 = data["Id1"].asUInt();
    id2 = data["Id2"].asUInt();
    weight = data["Weight"].asDouble();
}

void Edge::export_json(Json::Value& data)
{
    data["Id1"] = (unsigned int)(id1);
    data["Id2"] = (unsigned int)(id2);
    data["Weight"] = weight;
}

Graph::Graph(Json::Value& data)
{
    import_json(data);
}

void Graph::import_json(Json::Value& data)
{
    Json::Value vertices_data = data["Vertices"];
    Json::Value edges_data = data["Edges"];

    for (unsigned int i = 0; i < vertices_data.size(); ++i) {
        vertices.push_back(Vertex(vertices_data[i]));
    }

    for (unsigned int i = 0; i < edges_data.size(); ++i) {
        edges.push_back(Edge(edges_data[i]));
    }
}

void Graph::export_json(Json::Value& data)
{
    // create Vertex json array and assign to "Vertices" 
    Json::Value vertices_data(Json::arrayValue);
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        Json::Value vertex_data;
        vertices[i].export_json(vertex_data);
        vertices_data[i] = vertex_data;
    }
    data["Vertices"] = vertices_data;

    // create Edge json array and assign to "Edges" 
    Json::Value edges_data(Json::arrayValue);
    for (unsigned int i = 0; i < edges.size(); ++i) {
        Json::Value edge_data;
        edges[i].export_json(edge_data);
        edges_data[i] = edge_data;
    }
    data["Edges"] = edges_data;
}

}

