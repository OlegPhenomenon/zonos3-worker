/* Stand-in for libnccl.so.2 (383 MB). libggml-cuda links NCCL only for multi-GPU tensor parallelism
 * (ggml_backend_cuda_comm_init_nccl); every worker has exactly one GPU, so none of these is ever called. Linking the
 * real library only to satisfy the dynamic loader would cost ~290 MB of image. If one is called anyway the answer is
 * an error, never a silent success. */
#include <stddef.h>
#include <stdio.h>

typedef int ncclResult_t;
enum { ncclInvalidUsage = 5 };

static ncclResult_t unavailable(const char *fn) {
    fprintf(stderr, "libnccl stub: %s called, but this single-GPU image ships no NCCL\n", fn);
    return ncclInvalidUsage;
}

ncclResult_t ncclCommInitAll(void *comms, int ndev, const int *devlist) { (void)comms; (void)ndev; (void)devlist; return unavailable("ncclCommInitAll"); }
ncclResult_t ncclCommDestroy(void *comm) { (void)comm; return unavailable("ncclCommDestroy"); }
ncclResult_t ncclGroupStart(void) { return unavailable("ncclGroupStart"); }
ncclResult_t ncclGroupEnd(void) { return unavailable("ncclGroupEnd"); }
ncclResult_t ncclAllReduce(const void *send, void *recv, size_t count, int type, int op, void *comm, void *stream) {
    (void)send; (void)recv; (void)count; (void)type; (void)op; (void)comm; (void)stream;
    return unavailable("ncclAllReduce");
}
const char *ncclGetErrorString(ncclResult_t result) { (void)result; return "NCCL is not included in this image"; }
