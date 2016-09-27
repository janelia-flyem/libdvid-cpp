/*!
 * This file provides algorithms to quickly fetch several smaller
 * pieces of data from DVID in a more efficient way.  To this end,
 * it supports the ability to fetch data in parallel.  If threading
 * is enabled, the DVID backend should ideally be a distributed one.
 *
 * \author Stephen Plaza (plazas@janelia.hhmi.org)
*/
#ifndef THREADEDFETCH 
#define THREADEDFETCH

// TODO: implement a copy constructor for DVIDNodeService

#include "DVIDNodeService.h"
#include <boost/thread/thread.hpp>
#include <boost/asio/io_service.hpp>
#include <tr1/unordered_map>

namespace libdvid {


#define MINPOOLSIZE 16

/*!
 * Thread pool singleton for threading resources.
*/
class DVIDThreadPool {
  public:
    static DVIDThreadPool* get_pool()
    {
        static DVIDThreadPool pool;
        return &pool;
    }

    template <typename TTask>
    void add_task(TTask task)
    {
        io_service_.dispatch(task);
    }
    ~DVIDThreadPool()
    {
        delete work_ctrl_;
    }

  private:
    DVIDThreadPool()
    {
        work_ctrl_ = new boost::asio::io_service::work(io_service_);
        int workers = boost::thread::hardware_concurrency() * 2;
        workers = workers < MINPOOLSIZE ? MINPOOLSIZE : workers;

        for (int i = 0; i < workers; ++i) {
            threads_.create_thread(boost::bind(&boost::asio::io_service::run, &io_service_));
        }
    }

    boost::asio::io_service io_service_;
    boost::thread_group threads_;
    boost::asio::io_service::work *work_ctrl_;

};

/*!
 * Singleton for DVIDNode resources.
*/ 
class DVIDNodePool {
  public:
    static DVIDNodePool* get_pool()
    {
        static DVIDNodePool pool;
        return &pool;
    }

    std::tr1::unordered_map<std::string, std::vector<boost::shared_ptr<DVIDNodeService> > > dvidnodes; 

  private:
    DVIDNodePool()  { }
};

/*!
 * Fetches all the grayscale blocks that intersect the body id in the specified
 * label volume.  If threading is enabled, multiple requests will be done
 * simultaneously.  This call tries to minimize the number of http requests
 * by asking for contiguous chunks that include the necessary blocks.
 * \param service name of dvid node service
 * \param labelvol_name name of label volume with body id
 * \param grayscale_name name of grayscale data instance
 * \param num_threads number of threads used in the fetch.
 * \param use_blocks if true uses block interface instead of raw ND
 * \param request_efficiency how requests are packaged (0: 1 at a time, 1: X contig)
 * \return array of blocks matrix order (X = column, Y = row, Z=slice)
*/
std::vector<BinaryDataPtr> get_body_blocks(DVIDNodeService& service,
        std::string labelvol_name, std::string grayscale_name, uint64 bodyid,
        int num_threads = 1, bool use_blocks = false,
        int request_efficiency = 1);

/*!
 * Fetches all the label blocks that intersect the body id in the specified
 * label volume.  If threading is enabled, multiple requests will be done
 * simultaneously.  This call tries to minimize the number of http requests
 * by asking for contiguous chunks that include the necessary blocks.
 * \param service name of dvid node service
 * \param labelvol_name name of label volume with body id
 * \param labelsname name of labels data instance
 * \param spans X runs that make up the volume
 * \param num_threads number of threads used in the fetch.
 * \return array of blocks matrix order (X = column, Y = row, Z=slice)
*/
std::vector<BinaryDataPtr> get_body_labelblocks(DVIDNodeService& service,
        std::string labelvol_name, uint64 bodyid, std::string labelsname,
        std::vector<std::vector<int> >& spans, int num_threads = 2);

/*!
 * Write label blocks back to DVID at the specified spans.
 * \param service name of dvid node service
 * \param labelsname name of labels data instance
 * \param blocks list of label blocks to be written into DVID
 * \param spans X runs that make up the volume
 * \param num_threads number of threads used in the fetch.
*/
void put_labelblocks(DVIDNodeService& service, std::string labelsname,
        const std::vector<BinaryDataPtr>& blocks,
        std::vector<std::vector<int> >& spans, int num_threads = 2);


/*
 * Fetches all tile slices requested in parallel.
 * \param service name of dvid node service
 * \param orientation specify XY, YZ, or XZ
 * \param scaling specify zoom level (1=max res)
 * \param tile_locs_array e.g., X,Y,Z location of tile (X and Y are in block coordinates)
 * \param num_threads num_threads to use (0 means use as many as tiles)
 * \return byte buffer array with order the same as tiles requested
*/
std::vector<BinaryDataPtr> get_tile_array_binary(DVIDNodeService& service,
        std::string datatype_instance, Slice2D orientation, unsigned int scaling,
        const std::vector<std::vector<int> >& tile_locs_array, int num_threads=0);

}

#endif
