#ifndef DVIDALGORITHMS_H
#define DVIDALGORITHMS_H

#include "DVIDNodeService.h"

namespace libdvid {

/*!
 * Copies a body from node1 to node2.
 * \param node1 node with src body
 * \param node2 node with dest body
 * \param src_labels name of src labels
 * \param src_labelvol name of src labelvol
 * \param dest_labels name of dest labels
 * \param num_threads threads used in operation
*/
void copy_paste_body(DVIDNodeService& node1, DVIDNodeService& node2,
       uint64 src, uint64 dest, std::string src_labels,
       std::string src_labelvol, std::string dest_labels, int num_threads=1); 

}

#endif
