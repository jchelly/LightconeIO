#!/bin/env python

import socket
hostname = socket.gethostname()

try:
    import psutil
except ImportError:
    psutil = None


def report_usage(comm):
    """
    Report memory use of processes in the specified communicator
    """

    if psutil is None:
        return
    
    GB = 1024 ** 3

    # Get information about this node
    mem = psutil.virtual_memory()
    total_mem_gb = comm.allgather(mem.total / GB)
    free_mem_gb = comm.allgather(mem.available / GB)

    # Get information about this process
    process = psutil.Process()
    rss_gb  = comm.allgather(process.memory_info().rss / GB)

    # Report min and max
    if comm.Get_rank() == 0:
        min_free = min(free_mem_gb)
        max_free = max(free_mem_gb)
        min_rss  = min(rss_gb)
        max_rss  = max(rss_gb)
        print(f"Node free memory: min={min_free:.2f}GB, max={max_free:.2f}GB, process RSS: min={min_rss:.2f}GB, max={max_rss:.2f}GB")

        
if __name__ == "__main__":

    from mpi4py import MPI
    report_usage(MPI.COMM_WORLD)
