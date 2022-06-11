
void vta_gemm(int TA, int TB, int m_A, int k, int n_B, float ALPHA,
        float *A, int lda,
        int is_quant, 
        int8_t *A_q, float A_scale,
        float *B, int ldb,
        float BETA,
        float *C, int ldc, int layer_num, int shr,
        void* input_buf,
        void* weight_buf,
        void* output_buf);

void vta_gemm_cpu(int TA, int TB, int m_A, int k, int n_B, float ALPHA,
        float *A, int lda,
        int is_quant, 
        int8_t *A_q, float A_scale,
        float *B, int ldb,
        float BETA,
        float *C, int ldc, 
        int layer_num, int shr);