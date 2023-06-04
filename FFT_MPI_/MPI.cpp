#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
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
        if (i < j) std::swap(input[i], input[j]);
    }
    // 创建MPI环境
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int local_size = n / size;
    std::vector<std::complex<double>> local_input(local_size);
    // 使用消息传递进行数据划分和通信
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            MPI_Send(input.data() + i * local_size, local_size, MPI_DOUBLE_COMPLEX, i, 0, MPI_COMM_WORLD);
        }
        std::copy(input.begin(), input.begin() + local_size, local_input.begin());
    }
    else
    {
        MPI_Recv(local_input.data(), local_size, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    std::vector<std::complex<double>> local_output(local_size);
    // 蝴蝶运算
    for (int k = 2; k <= local_size; k <<= 1)
    {
        int m = k >> 1;
        std::complex<double> w_m(cos(PI / m), -sin(PI / m));

        for (int i = 0; i < local_size; i += k)
        {
            std::complex<double> w(1);
            for (int j = 0; j < m; j++)
            {
                std::complex<double> t = w * local_input[i + j + m];
                local_input[i + j + m] = local_input[i + j] - t;
                local_input[i + j] += t;
                w *= w_m;
            }
        }
    }
    // 使用消息传递进行结果收集
    if (rank == 0)
    {
        for (int i = 1; i < size; i++)
        {
            MPI_Recv(input.data() + i * local_size, local_size, MPI_DOUBLE_COMPLEX, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::copy(local_input.begin(), local_input.end(), input.begin());
    }
    else
    {
        MPI_Send(local_input.data(), local_size, MPI_DOUBLE_COMPLEX, 0, 0, MPI_COMM_WORLD);
    }
    // 串行处理剩余部分的数据
    for (int k = local_size * 2; k <= n; k <<= 1)
    {
        int m = k >> 1;
        std::complex<double> w_m(cos(PI / m), -sin(PI / m));

        for (int i = 0; i < n; i += k)
        {
            std::complex<double> w(1);
            for (int j = 0; j < m; j++)
            {
                std::complex<double> t = w * input[i + j + m];
                input[i + j + m] = input[i + j] - t;
                input[i + j] += t;
                w *= w_m;
            }
        }
    }
}



// mpi 运行命令 mpiexec -n 4 FFT_MPI.exe
int main(int argc, char* argv[])
{
    int rank;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream fi("fft_8388608.txt");
    vector<double> data;
    string read_temp;
    while (fi.good())
    {
        getline(fi, read_temp);
        data.push_back(stod(read_temp));
    }
    fi.close();

    vector<complex<double>> input2(data.size());
    for (size_t i = 0; i < data.size(); i++)
    {
        input2[i] = complex<double>(data[i], 0);
    }

    auto t1 = Clock::now();
    fft_mpi(input2);

    // Output
    if (rank == 0)
    {
        auto t2 = Clock::now();
        cout << "fft_mpi cost: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e+6 << " ms.\n";
        //ofstream fo;
        //fo.open("fft_mpi_result.txt", ios::out);
        //for (int i = 0; i < data.size(); i++)
        //{
        //    fo << '(' << input2[i].real() << ',' << input2[i].imag() << ')' << endl;
        //}
        //fo.close();
    }

    MPI_Finalize();
    return 0;
}





