#ifndef IMGPROCTEST
#define IMGPROCTEST
#define DATA_WIDTH 8
#define IMG_SIZE_0 320
#define IMG_SIZE_1 240
#define IMG_SIZE IMG_SIZE_0*IMG_SIZE_1

void ImgProcTest ( ac_fixed<DATA_WIDTH,DATA_WIDTH,false,AC_RND_INF,AC_SAT> img_in[IMG_SIZE],
                  ac_fixed<DATA_WIDTH,DATA_WIDTH,false,AC_RND_INF,AC_SAT> img_out[IMG_SIZE]
                  );
#endif
