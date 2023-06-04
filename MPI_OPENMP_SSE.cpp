#include <iostream>
#include <fstream>
#include <immintrin.h>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include<omp.h>
#include "mpi.h"

using namespace std;
using namespace chrono;

#define PI 3.1415926535897932384626433832
typedef std::chrono::high_resolution_clock Clock;

void fft_mpi(std::vector<std::complex<double>>& input)
{

    int n = input.size();
    // 数据重排
    for (int i = 1, j = 0; i < n; i++)
    {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) swap(input[i], input[j]);
    }
    // 创建MPI环境
    int rank, size;
    // MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<std::complex<double>> local_input(n / size);
    MPI_Scatter(input.data(), n / size, MPI_DOUBLE_COMPLEX, local_input.data(), n / size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
    std::vector<complex<double>> local_output(n / size);
// 蝴蝶计算
#pragma omp parallel num_threads(4)
    {
        for (int k = 2; k <= n / size; k <<= 1)
        {
            int m = k >> 1;
            complex<double> w_m(cos(PI / m), -sin(PI / m));
#pragma omp for 
            for (int i = 0; i < n / size; i += k)
            {
                for (int j = 0; j < m; j++)
                {
                    __m128d wr = _mm_set1_pd(cos(j * PI / m));
                    __m128d wi = _mm_set_pd(-sin(j * PI / m), sin(j * PI / m));
                    __m128d o = _mm_load_pd((double*)&local_input[i + j + m]);	// odd a|b
                    __m128d e = _mm_load_pd((double*)&local_input[i + j]);	// even
                    wr = _mm_mul_pd(o, wr);	// a*c|b*c
                    __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                    wi = _mm_mul_pd(n1, wi);
                    n1 = _mm_add_pd(wr, wi);
                    wr = _mm_add_pd(e, n1);
                    wi = _mm_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm_store_pd((double*)&local_input[i + j + m], wi);
                    // input[i + j] += t;
                    _mm_store_pd((double*)&local_input[i + j], wr);
                }
            }
        }

    }
    // 收集每个进程的局部结果
    MPI_Gather(local_input.data(), n / size, MPI_DOUBLE_COMPLEX, input.data(), n / size, MPI_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);

#pragma omp parallel num_threads(4)
    {

        for (int k = n / size * 2; k <= n; k <<= 1)
        {
            int m = k >> 1;
            complex<double> w_m(cos(PI / m), -sin(PI / m));

#pragma omp for 
            for (int i = 0; i < n; i += k)
            {
                for (int j = 0; j < m; j++)
                {
                    // complex<double> t = w * input[i + j + m];
                    // compute t
                    __m128d wr = _mm_set1_pd(cos(j * PI / m));
                    __m128d wi = _mm_set_pd(-sin(j * PI / m), sin(j * PI / m));
                    __m128d o = _mm_load_pd((double*)&input[i + j + m]);	// odd a|b
                    __m128d e = _mm_load_pd((double*)&input[i + j]);	// even
                    wr = _mm_mul_pd(o, wr);	// a*c|b*c
                    __m128d n1 = _mm_shuffle_pd(o, o, _MM_SHUFFLE2(0, 1));
                    wi = _mm_mul_pd(n1, wi);
                    n1 = _mm_add_pd(wr, wi);
                    wr = _mm_add_pd(e, n1);
                    wi = _mm_sub_pd(e, n1);
                    // input[i + j + m] = input[i + j] - t;
                    _mm_store_pd((double*)&input[i + j + m], wi);
                    // input[i + j] += t;
                    _mm_store_pd((double*)&input[i + j], wr);
                }
            }
        }
    }
}

// mpi 运行命令 mpiexec -n 4 FFT_MPI_OPENMP.exe
int main(int argc, char* argv[])
{
    int rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // 1024 524288 1048576 2097152 4194304 8388608 16777216
    ifstream fi("fft_8388608.txt");
    vector<double> data;
    string read_temp;
    while (fi.good())
    {
        getline(fi, read_temp);
        data.push_back(stod(read_temp));
    }
    vector<complex<double>> input2(data.size());
    for (size_t i = 0; i < data.size(); i++)
    {
        input2[i] = complex<double>(data[i], 0);
    }
    auto t1 = Clock::now();
    fft_mpi(input2);
    // 输出
    if (rank == 0)
    {
        auto t2 = Clock::now();// 计时结束
        cout << "fft_mos cost : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";
    }
    fi.close();
    MPI_Finalize();
    return 0;
}

