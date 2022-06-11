#include <iostream>
#include <ctime>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <vector>
#include <math.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include "vta/runtime.h"
#include "vta/data_buf.h"

#include <time.h>

#define VTA_COHERENT_ACCESSES true

#define GEMM_VAL 2

#define INP_IDX 16
#define WEIGHT_IDX 16
#define OUT_IDX 16


// using namespace std;
// using namespace vta;

struct timespec u_time;
double get_time(){
  clock_gettime(CLOCK_REALTIME, &u_time);
  return (u_time.tv_sec) + (u_time.tv_nsec) * 1e-9;
}

struct meta{
  int M;
  int K;
  int N;
  int MIDX;
  int NIDX;
  int K2048;
  int SHR;
};

int gemm_core(void* par, VTACommandHandle cmdh)
{
  // loop ->               extent,                                 dst,                       src,                   wgt
  // VTAUopPush(handle, mode, reset,  
  //                                                            dstidx,                    srcidx,                wgtidx, 
  //           opcode, useimm, immval)
  struct meta* p = (struct meta*)(par);
  VTAUopLoopBegin(cmdh, (p->K2048/16),                                   0,                        16,                     1);
  VTAUopLoopBegin(cmdh,        16,                                   1,                         1,                     0);
  VTAUopPush(cmdh,0,0,             16+(p->MIDX*(p->N/16) + p->NIDX)*16, 16, 16,  0,0,0);
  VTAUopLoopEnd(cmdh);
  VTAUopLoopEnd(cmdh);
  return 0;
}

int finit_gemmgemm(void* par, VTACommandHandle cmdh )
{
  struct meta* p = (struct meta*)(par);
  int init_num = p->N/16 * p->M / 16;
  VTAUopLoopBegin(cmdh, 16,1,0,0);
  for(int i=0; i<init_num; i++){
    VTAUopPush(cmdh,0,1,INP_IDX+16*i,0,0,0,0,0);
  }
  // VTAUopPush(cmdh,0,1,INP_IDX,INP_IDX,WEIGHT_IDX,0,0,0);
  VTAUopLoopEnd(cmdh);
  return 0;
}

int alu_max(void *par, VTACommandHandle cmdh) {
  struct meta* p = (struct meta*)(par);
  int init_num = p->N/16 * p->M/16;
  VTAUopLoopBegin(cmdh, 16, 1, 1, 0);
  for(int i=0; i<init_num; i++){
    VTAUopPush(cmdh, 1, 0, INP_IDX+16*i, 0, 0, 1, 1, -128);
  }
  VTAUopLoopEnd(cmdh);
  return 0;
}
int alu_min(void *par, VTACommandHandle cmdh) {
  struct meta* p = (struct meta*)(par);
  int init_num = p->N/16 * p->M/16;
  VTAUopLoopBegin(cmdh, 16, 1, 1, 0);
  for(int i=0; i<init_num; i++){
    VTAUopPush(cmdh, 1, 0, INP_IDX+16*i, 0, 0, 0, 1, 127);
  }
  VTAUopLoopEnd(cmdh);
  return 0;
}
int alu_shr(void *par,VTACommandHandle cmdh) {
  struct meta* p = (struct meta*)(par);
  int init_num = p->N/16 * p->M/16;
  VTAUopLoopBegin(cmdh, 16, 1, 1, 0);
  for(int i=0; i<init_num; i++){
    VTAUopPush(cmdh, 1, 0, INP_IDX+16*i, 0, 0, 3, 1, p->SHR);
  }
  VTAUopLoopEnd(cmdh);
  return 0;
}

void vta_gemm_cpu(int TA, int TB, int m_A, int k, int n_B, float ALPHA,
        float *A, int lda,
        int is_quant, 
        int8_t *A_q, float A_scale,
        float *B, int ldb,
        float BETA,
        float *C, int ldc, int layer_num, int shr)
{
  double latency = get_time();  
  int m = n_B;
  int n = m_A;
  int input_size = m*k;
  int weight_size = k*n;
  int output_size = m*n;

  float rng = 127.f;
  // shr = 10;

  int8_t *input  = (int8_t*)malloc(input_size);
  int8_t *weight = A_q;//(int8_t*)malloc(weight_size);
  int8_t *output = (int8_t*)malloc(output_size);

  int8_t *input_temp  = (int8_t*)malloc(input_size);
  // int8_t *weight_temp = (int8_t*)malloc(weight_size);
  int8_t *output_temp = (int8_t*)malloc(output_size);
  
  // float maxA = 0.f;
  // float maxA_pos = 0.f;
  // float maxA_neg = 0.f;
  
  // for(int i = 0; i < n; i++){
  //   for(int j = 0; j < k; j++){
  //     // if(fabs(A[i*k+j]) > maxA) maxA = fabs(A[i*k+j]);
      
  //     if(A[i*k+j] > 0.0){
  //       if(A[i*k+j] > maxA_pos)
  //         maxA_pos = A[i*k+j];
  //     }else{
  //       if(A[i*k+j] < maxA_neg)
  //         maxA_neg = A[i*k+j];
  //     }
  //   }
  // }
  float maxB = 0.f;
  float maxB_pos = 0.f;
  float maxB_neg = 0.f;

  for(int i = 0; i < k; i++){
    for(int j = 0; j < m; j++){
      // if(fabs(B[i*n+j]) > maxB) maxB = fabs(B[i*n+j]);
      if(B[i*m+j] > 0.0){
        if(B[i*m+j] > maxB_pos)
          maxB_pos = B[i*m+j];
      }else{
        if(B[i*m+j] < maxB_neg)
          maxB_neg = B[i*m+j];
      }
    }
  }
  float div=1.f;
  // maxA = (fabs(maxA_neg) > fabs(maxA_pos)) ? 
  //        (fabs(maxA_neg) - fabs(maxA_pos))/div +fabs(maxA_pos) : 
  //        (fabs(maxA_pos) - fabs(maxA_neg))/div +fabs(maxA_neg) ;
    
  maxB = (fabs(maxB_neg) > fabs(maxB_pos)) ? 
         (fabs(maxB_neg) - fabs(maxB_pos))/div +fabs(maxB_pos) : 
         (fabs(maxB_pos) - fabs(maxB_neg))/div +fabs(maxB_neg) ;

  float scale_we = A_scale;//maxA/rng;
  float scale_in = maxB/rng;
  
  // float scale_we = ceil(log(maxA/rng)/log(2.f))*ceil(log(maxA/rng)/log(2.f));
  // float scale_in = ceil(log(maxB/rng)/log(2.f))*ceil(log(maxB/rng)/log(2.f));//log(maxB/rng)/log(2)*log(maxB/rng)/log(2);
  
  printf("scale in : %f scale we : %f ",scale_in, scale_we);
  // printf("%d layer A max = %f B max = %f shr = %d ",layer_num, maxA, maxB, shr);
  // int shift = 6;
  // shr = 9;
  // printf("hello log? %f \n",log(2.0f)/log(2.0f));
  for(int i = 0; i<k; i++){
    for(int j = 0; j<m; j++){
      // input_temp[j*k+i] = (int8_t) (B[i*m+j] / maxB * rng);
      // int temp = B[i*m+j] / maxB * rng ;
      int temp = B[i*m+j] / scale_in;
      if(temp > 127) temp = 127;
      if(temp < -128) temp = -128;
      input_temp[j*k+i] = temp;
    }
  }
  // for(int i = 0; i<weight_size; i++){
  //   // weight_temp[i] = (int8_t) (A[i] / maxA * rng);
  //   // int temp = A[i] / maxA * rng;
  //   int temp = A[i] / scale_we;
  //   if(temp > 127) temp = 127;
  //   if(temp < -128) temp = -128;
  //   weight_temp[i] = temp;
  // }

  

  int max = 0;
  int out2[output_size] = {0};
  for(int i = 0; i< m; i++){
    for(int j = 0; j < n; j++){
      int summ = 0;
      for(int z = 0; z < k; z++){
        summ += (int)input_temp[i*k + z]*(int)weight[j*k + z];
      }
      if(abs(summ) > max){
        max = summ;
      }
      out2[i*n+j] = summ;
    }
  }
  // int shift = 6;
  int counter = 0;
  float err_min = 10000000000000.f;
  int err_min_shr;
  // shr = (int)(log(abs(max)>>7)/log(2))+3;
  // printf("! max %d , abs(max)/128 : %d , shr %d !", max , abs(max)>>7, shr);
  // fflush(stdout);


for(int cc = 0; cc < 7; cc++){

  for(int i = 0; i< m; i++){
    for(int j = 0; j < n; j++){
      int summ = out2[i*n + j];
      summ = summ >> shr;
      if(summ > 127)  summ = 127;
      if(summ < -128) summ = -128;
      output_temp[i*n+j] = (int8_t)summ;
    }
  }
  // m_A => N
  // n_B => M
//   float ref[output_size] = {0.0f};
// #pragma omp parallel for
//   for(int i = 0; i< m; i++){
//     for(int j = 0; j < n; j++){
//       float summ = 0.0f;
//       for(int z = 0; z < k; z++){
//         summ += A[j*k+z] * B[z*m+i];
//       }
//       ref[j*m+i] = summ;
//     }
//   }


  float err_acc = 0.0f;
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++ ){
      C[j*m+i] = (float)(output_temp[i*n+j] << shr) * scale_we * scale_in;
      // err_acc += fabs(ref[j*m+i] - C[j*m+i]);
      // C[j*m+i] = ref[j*m+i];
    }
  }
  
  

  // printf("%d layer A max = ( %f ~ [%f] ~ %f ) B max = ( %f ~ [%f] ~ %f) shr = %d \n",layer_num, maxA_neg, maxA, maxA_pos, maxB_neg, maxB, maxB_pos, shr);
  printf("%d layer A max = ( precomputed ) B max = ( %f ~ [%f] ~ %f) shr = %d ",layer_num, maxB_neg, maxB, maxB_pos, shr);
  // printf("max %d / log2 = %d\n",max, (int)(log(fabs((float)max)) ));
  // printf("err %f ... \n",err_acc);
  
  break;
  
  if(cc == 0){ 
    err_min = err_acc; 
    err_min_shr = shr;
    shr -= 5; 
    printf("set min : %f (shr=%d)",err_min,err_min_shr);
  }
  shr++;
  if(err_acc < err_min){
    err_min = err_acc;
    err_min_shr = shr-1;
    printf("min : %f (shr=%d)",err_min,err_min_shr);
  }
  if(cc == 5){
    shr = err_min_shr;
  }

}
  // xlnk_reset();
  // /*     K
  //  ┌  ─  ┬  ─  ┐
  //  │  1  │  2  │
  // M├  ─  ┼  ─  ┫
  //  │  3  │  4  │ M/16
  //  └  ─  ┻  ─  ┘
  //          K/16
  // */
  // //M/16,K/16,16,16
  // for(int i = 0 ; i < m/16; i++){
  //   for(int j = 0 ; j < k/16; j++) {
  //     for(int r = 0; r < 16; r++){
  //       for(int s = 0; s < 16; s++){
  //         int tmp_idx = (i*16 + r)*k + (j*16 + s);
  //         int inp_idx = i*k/16*16*16 + j*16*16 + r*16+ s;
  //         input[inp_idx] = input_temp[tmp_idx];
  //       }
  //     }
  //   }
  // }
  // //N/16,K/16,16,16
  // for(int i = 0 ; i < n/16; i++){
  //   for(int j = 0 ; j < k/16; j++){
  //     for(int r = 0; r < 16; r++){
  //       for(int s = 0; s < 16; s++){
  //         int tmp_idx = (i*16 + r)*k + (j*16 + s);
  //         int wgt_idx = i*k/16*16*16 + j*16*16 + r*16+ s;
  //         weight[wgt_idx] = weight_temp[tmp_idx];
  //       }
  //     }
  //   }
  // }

  // void* input_buf  = VTABufferAlloc( input_size );
  // void* weight_buf = VTABufferAlloc( weight_size);
  // void* output_buf = VTABufferAlloc( output_size);
  
  // VTABufferCopy(input, 0, input_buf, 0, m*k, 1);
  // VTABufferCopy(weight, 0, weight_buf, 0, n*k, 1);

  // {


  //   VTACommandHandle vtaCmdH{nullptr};
  //   vtaCmdH = VTATLSCommandHandle(EVTA_ID);

  //   // VTASetDebugMode(vtaCmdH, VTA_DEBUG_DUMP_INSN | VTA_DEBUG_DUMP_UOP);




  //   int k2048_loop = (k-1)/2048 + 1;
  //   int m_div_loop = (m/16*n/16)/128;
  //   int m_size;
  //   if(m_div_loop == 0){
  //     m_size = m;
  //   }else{
  //     m_size = m / m_div_loop;
  //   }
  //   m_div_loop += 1;    
  //   // uop kernel handle
  //   void *uop_init[2];
  //   uop_init[0] = nullptr;

  //   void *uop_shr[2];
  //   uop_shr[0] = nullptr;

  //   void *uop_max[2];
  //   uop_max[0] = nullptr;

  //   void *uop_min[2];
  //   uop_min[0] = nullptr;

  //   struct meta par;
  //   par.M = m_size;
  //   par.K = k;
  //   par.N = n;

  //   void*** uop = (void***)malloc((m_size/16*n/16*k2048_loop) * sizeof(size_t));
  //   for(int i =0; i<(m_size/16*n/16*k2048_loop); i++){
  //     uop[i] = (void**)malloc(2  * sizeof(size_t));
  //     uop[i][0] = nullptr;
  //   }

  //   for(int m_div = 0; m_div < m_div_loop; m_div++){

  //     // printf("mdivloop : %d m_div :%d m_size : %d\n",m_div_loop,m_div,m_size);
  //     //reset acc mem
  //     VTAPushGEMMOp(vtaCmdH,uop_init, &finit_gemmgemm, &par, 0);

  //     VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);
  //     // run gemm
  //     for(int i = 0; i < m_size/16; i++){
        
  //       for(int k_div = 0; k_div < k2048_loop; k_div++){  
  //         int k2048 = 2048;
  //         if(k%2048 != 0 && k2048_loop == k_div+1){
  //           k2048 = k % 2048;
  //         }

  //         // VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);
  //         int in_16_K     = 16 * k/16;
  //         int in_16_K2048 = 16 * k2048/16;
  //         VTALoadBuffer2D(vtaCmdH, input_buf,  m_div*m_size*in_16_K/16 + in_16_K*i + k_div*2048,  in_16_K2048,  1, 1, 0, 0, 0, 0, INP_IDX, 2);
  //         // printf("in offset = %d\n",in_16_K*i + k_div*2048*16);
  //         // VTADepPush(vtaCmdH, 1, 2);    VTADepPop(vtaCmdH, 1, 2);

  //         for(int j = 0; j < n/16; j++){
  //           par.MIDX = i;
  //           par.NIDX = j;
  //           // printf("i, j = %d, %d\n", i,j);
  //           // VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);
  //           // int wg_N_K_d1616 = k/16 * n*16;
  //           int wg_16_K     = k/16;
  //           int wg_16_K2048 = k2048/16;
  //           par.K2048 = k2048;
  //           // printf("k2048 %d / range %d?\n",par.K2048,wg_16_K2048);
  //           VTALoadBuffer2D(vtaCmdH, weight_buf, wg_16_K*j + 2048/16*k_div,  wg_16_K2048, 1, 1, 0, 0, 0, 0, WEIGHT_IDX, 1);
  //           // printf("we offset = %d\n",wg_16_K*j + 2048*k_div);
  //           VTADepPush(vtaCmdH, 1, 2);    VTADepPop(vtaCmdH, 1, 2);
            
  //           VTAPushGEMMOp(vtaCmdH, uop[k_div*m_size/16*n/16 + i*n/16 +j], &gemm_core, &par, 0);
  //           // if(k2048 < 2048){
  //           //   printf("2048 routine\n");
  //           VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);  
  //           // }else{
  //           //   VTAPushGEMMOp(vtaCmdH, uop[k_div*m/16*n/16 + i*n/16 +j], &gemm2_2, &par, 0);
  //           // }
  //           // VTAPushGEMMOp(vtaCmdH, uopHandle2, &gemm2_2, &par, 0);
            
  //         }
  //       }
  //     }
  //     VTADepPush(vtaCmdH, 1, 2);    VTADepPop(vtaCmdH, 1, 2);
  //     par.SHR = shr;
  //     VTAPushALUOp(vtaCmdH, uop_shr, &alu_shr, &par, 0);
  //     VTAPushALUOp(vtaCmdH, uop_max, &alu_max, &par, 0);
  //     VTAPushALUOp(vtaCmdH, uop_min, &alu_min, &par, 0);

  //     VTADepPush(vtaCmdH, 2, 3);    VTADepPop(vtaCmdH, 2, 3);

  //     // store stage-1 result
  //     int ou_M_N_d16 = m_size * n/16;
  //     VTAStoreBuffer2D(vtaCmdH, OUT_IDX, 4, output_buf, m_div*m_size*n/16, ou_M_N_d16, 1, 1);
  //     VTADepPush(vtaCmdH, 3, 2);    VTADepPop(vtaCmdH, 3, 2);  
  //   }

  //   VTASynchronize(vtaCmdH, 1 << 31);
  // }


  // VTABufferCopy(output_buf, 0, output, 0, output_size, 2);  // 1 MemCopyToHost



  // float deq_scale = 1.f / scale_we * 1.f / scale_in;
  // float max_c = 0.0f;
  // for(int i = 0; i < m; i++){
  //   for(int j = 0; j < n; j++){
  //     C[j*m + i] = (float)(output[i*n + j]<<shr) * deq_scale;
  //     if(fabs(C[j*m + i]) > max_c){
  //       max_c = fabs(C[j*m + i]);
  //     }
  //   }
  // }
  // printf(" max c  = %f\n", max_c);


  latency = get_time() - latency;
  double OPS = (double)(2.f*k-1.f);
  OPS = (double)(m*n)*OPS * 1e-9;
  // printf("%lf GOPS\n",OPS);
  printf("m k n : %d %d %d => %lf GOPS (time : %lf) ",m,k,n, OPS/latency, latency);

  // VTARuntimeShutdown(EVTA_ID);
  
  // VTABufferFree( input_buf );
  // VTABufferFree( output_buf);
  // VTABufferFree( weight_buf);
  free(input);
  // free(weight);
  free(input_temp);
  // free(weight_temp);
  free(output_temp);
  free(output);
}

// uop kernel handle
void *uop_init[2];
void *uop_shr[2];
void *uop_max[2];
void *uop_min[2];
extern VTACommandHandle vtaCmdH;

void vta_gemm(int TA, int TB, int m_A, int k, int n_B, float ALPHA,
        float *A, int lda,
        int is_quant, 
        int8_t *A_q, float A_scale,
        float *B, int ldb,
        float BETA,
        float *C, int ldc, int layer_num, int shr,
        void* input_buf,
        void* weight_buf,
        void* output_buf)
{
  
  int m = n_B;
  int n = m_A;
  int input_size = m*k;
  int weight_size = k*n;
  int output_size = m*n;

  float rng = 127.f;

  int8_t *input  = (int8_t*)malloc(input_size);
  // int8_t *weight = (int8_t*)malloc(weight_size);
  int8_t *output = (int8_t*)malloc(output_size);

  int8_t *input_temp  = (int8_t*)malloc(input_size);
  // int8_t *weight_temp = A_q;
  int8_t *weight = A_q;
  int8_t *output_temp = (int8_t*)malloc(output_size);

  float maxB = 0.f;
  float maxB_pos = 0.f;
  float maxB_neg = 0.f;

  for(int i = 0; i < k; i++){
    for(int j = 0; j < m; j++){
      // if(fabs(B[i*n+j]) > maxB) maxB = fabs(B[i*n+j]);
      if(B[i*m+j] > 0.0f){
        if(B[i*m+j] > maxB_pos)
          maxB_pos = B[i*m+j];
      }else{
        if(B[i*m+j] < maxB_neg)
          maxB_neg = B[i*m+j];
      }
    }
  }
  float div=1.f;
  // maxA = (fabs(maxA_neg) > fabs(maxA_pos)) ? 
  //        (fabs(maxA_neg) - fabs(maxA_pos))/div +fabs(maxA_pos) : 
  //        (fabs(maxA_pos) - fabs(maxA_neg))/div +fabs(maxA_neg) ;
    
  maxB = (fabs(maxB_neg) > fabs(maxB_pos)) ? 
         (fabs(maxB_neg) - fabs(maxB_pos))/div +fabs(maxB_pos) : 
         (fabs(maxB_pos) - fabs(maxB_neg))/div +fabs(maxB_neg) ;

  float scale_we = A_scale;//maxA/rng;
  float scale_in = maxB/rng;
  // scale_in = 0.005f;
  printf("scale in : %f scale we : %f ",scale_in, scale_we);
  // shr = ceil(log(1/(scale_we * scale_in*128))/log(2));
// #pragma omp parallel for
  for(int i = 0; i<k; i++){
    for(int j = 0; j<m; j++){
      int temp = B[i*m+j] / scale_in;
      if(temp > 127) temp = 127;
      if(temp < -128) temp = -128;
      input_temp[j*k+i] = (int8_t)temp;
    }
  }

  /*     K
   ┌  ─  ┬  ─  ┐
   │  1  │  2  │
  M├  ─  ┼  ─  ┫
   │  3  │  4  │ M/16
   └  ─  ┻  ─  ┘
           K/16
  */
  //M/16,K/16,16,16
#pragma omp parallel for
  for(int i = 0 ; i < m/16; i++){
    for(int j = 0 ; j < k/16; j++) {
      for(int r = 0; r < 16; r++){
        for(int s = 0; s < 16; s++){
          int tmp_idx = (i*16 + r)*k + (j*16 + s);
          int inp_idx = i*k/16*16*16 + j*16*16 + r*16+ s;
          input[inp_idx] = input_temp[tmp_idx];
        }
      }
    }
  }
  
  //N/16,K/16,16,16
  // #pragma omp parallel for
  // for(int i = 0 ; i < n/16; i++){
  //   for(int j = 0 ; j < k/16; j++){
  //     for(int r = 0; r < 16; r++){
  //       for(int s = 0; s < 16; s++){
  //         int tmp_idx = (i*16 + r)*k + (j*16 + s);
  //         int wgt_idx = i*k/16*16*16 + j*16*16 + r*16+ s;
  //         weight[wgt_idx] = weight_temp[tmp_idx];
  //       }
  //     }
  //   }
  // }

  // void* input_buf  = VTABufferAlloc( input_size );
  // void* weight_buf = VTABufferAlloc( weight_size);
  // void* output_buf = VTABufferAlloc( output_size);
  
  VTABufferCopy(input, 0, input_buf, 0, m*k, 1);
  VTABufferCopy(weight, 0, weight_buf, 0, n*k, 1);

  double latency = get_time();  

  VTAUopBufferReset(vtaCmdH);

  uop_init[0] = nullptr;
  uop_shr[0] = nullptr;
  uop_max[0] = nullptr;
  uop_min[0] = nullptr;


  {

    // VTASetDebugMode(vtaCmdH, VTA_DEBUG_DUMP_INSN | VTA_DEBUG_DUMP_UOP);


    int k2048_loop = (k-1)/2048 + 1;
    int m_div_loop = (m/16*n/16)/128;
    int m_size;
    if(m_div_loop == 0){
      m_size = m;
    }else{
      m_size = m / m_div_loop;
    }
    m_div_loop += 1;    

    struct meta par;
    par.M = m_size;
    par.K = k;
    par.N = n;

    void*** uop = (void***)malloc((m_size/16*n/16*k2048_loop) * sizeof(size_t));
    for(int i =0; i<(m_size/16*n/16*k2048_loop); i++){
      uop[i] = (void**)malloc(2  * sizeof(size_t));
      uop[i][0] = nullptr;
    }

    for(int m_div = 0; m_div < m_div_loop; m_div++){

      // printf("mdivloop : %d m_div :%d m_size : %d\n",m_div_loop,m_div,m_size);
      //reset acc mem
      VTAPushGEMMOp(vtaCmdH, uop_init, &finit_gemmgemm, &par, 0);

      VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);
      // run gemm
      for(int i = 0; i < m_size/16; i++){
        
        for(int k_div = 0; k_div < k2048_loop; k_div++){  
          int k2048 = 2048;
          if(k%2048 != 0 && k2048_loop == k_div+1){
            k2048 = k % 2048;
          }

          // VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);
          int in_16_K     = 16 * k/16;
          int in_16_K2048 = 16 * k2048/16;
          VTALoadBuffer2D(vtaCmdH, input_buf,  m_div*m_size*in_16_K/16 + in_16_K*i + k_div*2048,  in_16_K2048,  1, 1, 0, 0, 0, 0, INP_IDX, 2);
          // printf("in offset = %d\n",in_16_K*i + k_div*2048*16);
          // VTADepPush(vtaCmdH, 1, 2);    VTADepPop(vtaCmdH, 1, 2);

          for(int j = 0; j < n/16; j++){
            par.MIDX = i;
            par.NIDX = j;
            // printf("i, j = %d, %d\n", i,j);
            // VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);
            // int wg_N_K_d1616 = k/16 * n*16;
            int wg_16_K     = k/16;
            int wg_16_K2048 = k2048/16;
            par.K2048 = k2048;
            // printf("k2048 %d / range %d?\n",par.K2048,wg_16_K2048);
            VTALoadBuffer2D(vtaCmdH, weight_buf, wg_16_K*j + 2048/16*k_div,  wg_16_K2048, 1, 1, 0, 0, 0, 0, WEIGHT_IDX, 1);
            // printf("we offset = %d\n",wg_16_K*j + 2048*k_div);
            VTADepPush(vtaCmdH, 1, 2);    VTADepPop(vtaCmdH, 1, 2);
            
            VTAPushGEMMOp(vtaCmdH, uop[k_div*m_size/16*n/16 + i*n/16 +j], &gemm_core, &par, 0);
            // if(k2048 < 2048){
            //   printf("2048 routine\n");
            VTADepPush(vtaCmdH, 2, 1);    VTADepPop(vtaCmdH, 2, 1);  
            // }else{
            //   VTAPushGEMMOp(vtaCmdH, uop[k_div*m/16*n/16 + i*n/16 +j], &gemm2_2, &par, 0);
            // }
            // VTAPushGEMMOp(vtaCmdH, uopHandle2, &gemm2_2, &par, 0);
          }
        }
      }
      VTADepPush(vtaCmdH, 1, 2);    VTADepPop(vtaCmdH, 1, 2);
      par.SHR = shr;
      VTAPushALUOp(vtaCmdH, uop_shr, &alu_shr, &par, 0);
      VTAPushALUOp(vtaCmdH, uop_max, &alu_max, &par, 0);
      VTAPushALUOp(vtaCmdH, uop_min, &alu_min, &par, 0);

      VTADepPush(vtaCmdH, 2, 3);    VTADepPop(vtaCmdH, 2, 3);

      // store stage-1 result
      int ou_M_N_d16 = m_size * n/16;
      VTAStoreBuffer2D(vtaCmdH, OUT_IDX, 4, output_buf, m_div*m_size*n/16, ou_M_N_d16, 1, 1);
      VTADepPush(vtaCmdH, 3, 2);    VTADepPop(vtaCmdH, 3, 2);  
    }

    VTASynchronize(vtaCmdH, 1 << 31);
  }

  VTABufferCopy(output_buf, 0, output, 0, output_size, 2);  // 1 MemCopyToHost

  latency = get_time() - latency;

#pragma omp parallel for
  for(int i = 0 ; i < m/16; i++){
    for(int j = 0 ; j < n/16; j++) {
      for(int r = 0; r < 16; r++){
        for(int s = 0; s < 16; s++){
          int tmp_idx = i*n/16*16*16 + j*16*16 + r*16+ s;
          int out_idx = (i*16 + r)*n + (j*16 + s);
          output_temp[out_idx] = output[tmp_idx];
        }
      }
    }
  }
  
  // for(int i = 0; i<100; i++){
  //   printf("(%d)",output[i]);
  // }


  // float deq_scale = 1.f / scale_we * 1.f / scale_in;
  // float max_c = 0.0f;
#pragma omp parallel for
  for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
      C[j*m + i] = (float)((int)output_temp[i*n + j]<<shr) * scale_we * scale_in;
      // if(fabs(C[j*m + i]) > max_c){
      //   max_c = fabs(C[j*m + i]);
      // }
    }
  }
  // printf(" max c  = %f\n", max_c);  
  // printf("%d layer A max = ( precomputed ) B max = ( %f ~ [%f] ~ %f) shr = %d ",layer_num, maxB_neg, maxB, maxB_pos, shr);

  
  double OPS = (double)(2.f*k-1.f);
  OPS = (double)(m*n)*OPS * 1e-9;
  // printf("%lf GOPS\n",OPS);
  printf("m k n : %d %d %d => %lf GOPS (time : %lf) - ",m,k,n, OPS/latency, latency);

  free(input);
  // free(weight);
  free(output);
  free(input_temp);
  // free(weight_temp);
  free(output_temp);

  // VTABufferFree( input_buf );
  // VTABufferFree( output_buf);
  // VTABufferFree( weight_buf);
  

}

