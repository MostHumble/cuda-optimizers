#include <cuda_runtime.h>

// following pytorch docs:
// https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html

__global__ void adam_optimization_kernel(
    float* params,           
    const float* gradients, 
    float* first_moments,
    float* second_moments,
    float* max_second_moments,
    const float learning_rate,
    const float beta1,
    const float beta2,
    const float epsilon,
    const float weight_decay,
    const int t,
    const int size,
    const bool maximize,
    const bool use_amsgrad
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {

      float grad = maximize ? -gradients[i] : gradients[i];

        params[i] *= (1.0f - learning_rate * weight_decay);

        first_moments[i] = beta1 * first_moments[i] + 
                           (1.0f - beta1) * grad;

        second_moments[i] = beta2 * second_moments[i] + 
                            (1.0f - beta2) * grad * grad;

        float m_hat = first_moments[i] / (1.0f - powf(beta1, t));
        float v_hat = second_moments[i] / (1.0f - powf(beta2, t));

        if (use_amsgrad) {
            max_second_moments[i] = fmaxf(max_second_moments[i], v_hat);
            v_hat = max_second_moments[i];
        }

      float adaptive_lr = learning_rate * m_hat / 
                            (sqrtf(v_hat) + epsilon);

        params[i] -= adaptive_lr;
    }
}
